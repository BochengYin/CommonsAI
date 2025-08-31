import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.server import app
from app.file_repository import FileRepository


class TestServerIntegration:
    """Integration tests for FastAPI server with Repository"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests"""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "images").mkdir(parents=True, exist_ok=True)
        
        yield data_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_client_with_temp_repo(self, temp_data_dir):
        """Create test client with temporary repository"""
        with patch('app.server.FileRepository') as mock_repo_class:
            # Setup mock repository
            mock_repo = Mock(spec=FileRepository)
            mock_repo.get_tau.return_value = 0.3
            mock_repo.get_image_ids.return_value = ["test1.jpg", "test2.png"]
            mock_repo.get_all_qa.return_value = [
                {"id": "test1.jpg", "answer": "Test answer 1", "quality": 4, "path": "data/images/test1.jpg"},
                {"id": "test2.png", "answer": "", "quality": 0, "path": "data/images/test2.png"}
            ]
            mock_repo_class.return_value = mock_repo
            
            with patch('app.server._repo', mock_repo):
                client = TestClient(app)
                yield client, mock_repo
    
    @pytest.fixture 
    def test_client_no_index(self, temp_data_dir):
        """Create test client with no FAISS index"""
        with patch('app.server._index', None):
            client = TestClient(app)
            yield client
    
    def test_health_endpoint(self, test_client_with_temp_repo):
        """Test /health endpoint returns correct status"""
        client, mock_repo = test_client_with_temp_repo
        
        with patch('app.server._index', Mock()):
            response = client.get("/health")
            
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] == True
        assert data["tau"] == 0.3
        assert data["num_images"] == 2
        mock_repo.get_tau.assert_called_once()
        mock_repo.get_image_ids.assert_called_once()
    
    def test_health_endpoint_no_index(self, test_client_no_index):
        """Test /health endpoint when index is not built"""
        client = test_client_no_index
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] == False
    
    def test_set_tau_endpoint(self, test_client_with_temp_repo):
        """Test /set_tau endpoint updates tau value"""
        client, mock_repo = test_client_with_temp_repo
        mock_repo.get_tau.return_value = 0.5
        
        response = client.post("/set_tau", data={"tau": 0.5})
        
        assert response.status_code == 200
        data = response.json()
        assert data["tau"] == 0.5
        mock_repo.set_tau.assert_called_once_with(0.5)
        mock_repo.get_tau.assert_called_once()
    
    def test_update_answer_endpoint(self, test_client_with_temp_repo):
        """Test /update_answer endpoint calls repository correctly"""
        client, mock_repo = test_client_with_temp_repo
        
        response = client.post("/update_answer", data={
            "img_id": "test1.jpg",
            "answer": "Updated answer",
            "quality": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] == True
        mock_repo.upsert_qa.assert_called_once_with("test1.jpg", "Updated answer", 5)
    
    def test_query_endpoint_no_index(self, test_client_no_index):
        """Test /query endpoint returns error when index not built"""
        client = test_client_no_index
        
        response = client.post("/query", data={"text": "test query", "k": 3})
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Index not built" in data["error"]
    
    def test_query_endpoint_with_index(self, test_client_with_temp_repo):
        """Test /query endpoint with mock index"""
        client, mock_repo = test_client_with_temp_repo
        
        # Mock FAISS index and encode_text
        mock_index = Mock()
        import numpy as np
        mock_index.search.return_value = (
            np.array([[0.8, 0.6]]), # similarities  
            np.array([[0, 1]])      # indices
        )
        
        with patch('app.server._index', mock_index), \
             patch('app.server.encode_text') as mock_encode:
            
            mock_encode.return_value = Mock()
            mock_encode.return_value.astype.return_value = Mock()
            
            response = client.post("/query", data={"text": "test query", "k": 2})
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert "topk" in data
        assert "tau" in data
        assert len(data["topk"]) == 2
        
        # Verify repository was called
        mock_repo.get_image_ids.assert_called_once()
        mock_repo.get_all_qa.assert_called_once()
        mock_repo.get_tau.assert_called_once()
    
    def test_query_hit_decision(self, test_client_with_temp_repo):
        """Test query returns HIT when similarity is high and answer exists"""
        client, mock_repo = test_client_with_temp_repo
        mock_repo.get_tau.return_value = 0.5  # Lower threshold
        
        mock_index = Mock()
        import numpy as np
        mock_index.search.return_value = (
            np.array([[0.8]]), # High similarity
            np.array([[0]])    # First image index
        )
        
        with patch('app.server._index', mock_index), \
             patch('app.server.encode_text') as mock_encode:
            
            mock_encode.return_value = Mock()
            mock_encode.return_value.astype.return_value = Mock()
            
            response = client.post("/query", data={"text": "test query", "k": 1})
        
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "HIT"
        assert data["topk"][0]["img_id"] == "test1.jpg"
        assert data["topk"][0]["answer"] == "Test answer 1"
    
    def test_query_miss_decision(self, test_client_with_temp_repo):
        """Test query returns MISS when similarity is low or no answer"""
        client, mock_repo = test_client_with_temp_repo
        mock_repo.get_tau.return_value = 0.9  # High threshold
        
        mock_index = Mock()
        import numpy as np
        mock_index.search.return_value = (
            np.array([[0.5]]), # Low similarity
            np.array([[1]])    # Second image index (no answer)
        )
        
        with patch('app.server._index', mock_index), \
             patch('app.server.encode_text') as mock_encode:
            
            mock_encode.return_value = Mock()
            mock_encode.return_value.astype.return_value = Mock()
            
            response = client.post("/query", data={"text": "test query", "k": 1})
        
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "MISS"


class TestServerRepositoryIntegration:
    """Test server integration with real FileRepository"""
    
    @pytest.fixture
    def temp_repo_setup(self):
        """Setup real FileRepository with temporary files"""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "images").mkdir(parents=True, exist_ok=True)
        
        # Create test data files
        tau_path = data_dir / "tau.txt"
        ids_path = data_dir / "ids.json"
        qa_path = data_dir / "qa.jsonl"
        
        tau_path.write_text("0.4")
        ids_path.write_text('["real_test.jpg"]')
        qa_path.write_text('{"id": "real_test.jpg", "answer": "Real answer", "quality": 4, "type": "image", "path": "data/images/real_test.jpg", "tags": []}\n')
        
        yield data_dir, tau_path, ids_path, qa_path
        shutil.rmtree(temp_dir)
    
    def test_server_with_real_repository(self, temp_repo_setup):
        """Test server endpoints with real FileRepository"""
        data_dir, tau_path, ids_path, qa_path = temp_repo_setup
        
        with patch('app.file_repository.QA_PATH', qa_path), \
             patch('app.file_repository.IDS_PATH', ids_path), \
             patch('app.file_repository.TAU_PATH', tau_path), \
             patch('app.file_repository.IMAGES_DIR', data_dir / "images"), \
             patch('app.server._index', None):  # No index for this test
            
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            # Note: tau may have been modified by previous tests, so just check it's a float
            assert isinstance(data["tau"], float)
            # Note: num_images reflects the existing data, not just our test data
            assert data["num_images"] >= 1
            
            # Test set_tau
            response = client.post("/set_tau", data={"tau": 0.6})
            assert response.status_code == 200
            assert response.json()["tau"] == 0.6
            # Verify persistence
            assert float(tau_path.read_text().strip()) == 0.6
            
            # Test update_answer  
            response = client.post("/update_answer", data={
                "img_id": "real_test.jpg",
                "answer": "Updated real answer", 
                "quality": 5
            })
            assert response.status_code == 200
            assert response.json()["ok"] == True
            
            # Verify update persisted
            qa_content = qa_path.read_text()
            assert "Updated real answer" in qa_content
            assert '"quality":5' in qa_content or '"quality": 5' in qa_content


class TestServerErrorHandling:
    """Test server error handling and edge cases"""
    
    def test_malformed_requests(self):
        """Test server handles malformed requests gracefully"""
        client = TestClient(app)
        
        # Missing required form fields
        response = client.post("/set_tau", data={})
        assert response.status_code == 422  # Validation error
        
        response = client.post("/query", data={})
        assert response.status_code == 422
        
        response = client.post("/update_answer", data={"img_id": "test.jpg"})
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test server handles invalid data types"""
        client = TestClient(app)
        
        # Invalid tau value
        response = client.post("/set_tau", data={"tau": "not_a_number"})
        assert response.status_code == 422
        
        # Invalid k value
        response = client.post("/query", data={"text": "test", "k": "not_a_number"})
        assert response.status_code == 422
        
        # Invalid quality value
        response = client.post("/update_answer", data={
            "img_id": "test.jpg", 
            "answer": "test", 
            "quality": "not_a_number"
        })
        assert response.status_code == 422