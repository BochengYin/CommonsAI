import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch

from app.file_repository import FileRepository
from app.utils import ensure_dirs


class TestFileRepository:
    """Test FileRepository implementation"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create isolated temporary directory for each test"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repo_with_temp_paths(self, temp_dir):
        """Create FileRepository with temporary data paths"""
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock the paths in utils module
        with patch('app.file_repository.QA_PATH', data_dir / "qa.jsonl"), \
             patch('app.file_repository.IDS_PATH', data_dir / "ids.json"), \
             patch('app.file_repository.TAU_PATH', data_dir / "tau.txt"), \
             patch('app.file_repository.IMAGES_DIR', images_dir):
            
            repo = FileRepository()
            yield repo, data_dir
    
    def test_initialization(self, repo_with_temp_paths):
        """FileRepository should initialize without errors"""
        repo, data_dir = repo_with_temp_paths
        assert repo is not None
        assert repo._cached_ids is None
        assert repo._cached_qa is None
        assert repo._cached_tau is None
    
    def test_get_tau_default(self, repo_with_temp_paths):
        """get_tau should return default value when file doesn't exist"""
        repo, data_dir = repo_with_temp_paths
        tau = repo.get_tau()
        assert tau == 0.3
        assert isinstance(tau, float)
    
    def test_set_and_get_tau(self, repo_with_temp_paths):
        """set_tau should persist and get_tau should retrieve the value"""
        repo, data_dir = repo_with_temp_paths
        
        # Set new tau value
        new_tau = 0.75
        repo.set_tau(new_tau)
        
        # Should return updated value
        assert repo.get_tau() == new_tau
        
        # Should persist across instances
        new_repo = FileRepository()
        with patch('app.file_repository.TAU_PATH', data_dir / "tau.txt"):
            assert new_repo.get_tau() == new_tau
    
    def test_get_image_ids_empty(self, repo_with_temp_paths):
        """get_image_ids should return empty list when no data exists"""
        repo, data_dir = repo_with_temp_paths
        ids = repo.get_image_ids()
        assert ids == []
        assert isinstance(ids, list)
    
    def test_add_image_id(self, repo_with_temp_paths):
        """add_image_id should add and persist image IDs"""
        repo, data_dir = repo_with_temp_paths
        
        # Add first ID
        repo.add_image_id("test1.jpg")
        assert "test1.jpg" in repo.get_image_ids()
        
        # Add second ID  
        repo.add_image_id("test2.png")
        ids = repo.get_image_ids()
        assert len(ids) == 2
        assert "test1.jpg" in ids
        assert "test2.png" in ids
        
        # Should return copies to prevent modification
        ids[0] = "modified"
        original_ids = repo.get_image_ids()
        assert original_ids[0] != "modified"
    
    def test_get_all_qa_empty(self, repo_with_temp_paths):
        """get_all_qa should return empty list when no data exists"""
        repo, data_dir = repo_with_temp_paths
        qa_records = repo.get_all_qa()
        assert qa_records == []
        assert isinstance(qa_records, list)
    
    def test_get_qa_by_id_nonexistent(self, repo_with_temp_paths):
        """get_qa_by_id should return None for nonexistent IDs"""
        repo, data_dir = repo_with_temp_paths
        result = repo.get_qa_by_id("nonexistent.jpg")
        assert result is None
    
    def test_upsert_qa_new_record(self, repo_with_temp_paths):
        """upsert_qa should create new QA record"""
        repo, data_dir = repo_with_temp_paths
        
        img_id = "test.jpg"
        answer = "Test answer"
        quality = 4
        
        repo.upsert_qa(img_id, answer, quality)
        
        # Should be retrievable
        record = repo.get_qa_by_id(img_id)
        assert record is not None
        assert record["id"] == img_id
        assert record["answer"] == answer
        assert record["quality"] == quality
        assert record["type"] == "image"
        assert record["tags"] == []
        
        # Should appear in all records
        all_records = repo.get_all_qa()
        assert len(all_records) == 1
        assert all_records[0]["id"] == img_id
    
    def test_upsert_qa_update_existing(self, repo_with_temp_paths):
        """upsert_qa should update existing QA record"""
        repo, data_dir = repo_with_temp_paths
        
        img_id = "test.jpg"
        
        # Create initial record
        repo.upsert_qa(img_id, "Initial answer", 2)
        initial_record = repo.get_qa_by_id(img_id)
        assert initial_record["answer"] == "Initial answer"
        assert initial_record["quality"] == 2
        
        # Update the record
        repo.upsert_qa(img_id, "Updated answer", 5, tags=["updated"])
        updated_record = repo.get_qa_by_id(img_id)
        assert updated_record["answer"] == "Updated answer" 
        assert updated_record["quality"] == 5
        assert updated_record["tags"] == ["updated"]
        
        # Should still be only one record
        all_records = repo.get_all_qa()
        assert len(all_records) == 1
    
    def test_upsert_qa_with_custom_path_and_tags(self, repo_with_temp_paths):
        """upsert_qa should handle custom path and tags"""
        repo, data_dir = repo_with_temp_paths
        
        img_id = "custom.jpg"
        custom_path = "/custom/path/custom.jpg"
        custom_tags = ["tag1", "tag2"]
        
        repo.upsert_qa(img_id, "Custom answer", 3, custom_path, custom_tags)
        
        record = repo.get_qa_by_id(img_id)
        assert record["path"] == custom_path
        assert record["tags"] == custom_tags
    
    def test_refresh_clears_cache(self, repo_with_temp_paths):
        """refresh should clear all cached data"""
        repo, data_dir = repo_with_temp_paths
        
        # Populate cache by accessing data
        repo.get_tau()
        repo.get_image_ids()
        repo.get_all_qa()
        
        # Verify cache is populated
        assert repo._cached_tau is not None
        assert repo._cached_ids is not None
        assert repo._cached_qa is not None
        
        # Refresh should clear cache
        repo.refresh()
        assert repo._cached_tau is None
        assert repo._cached_ids is None
        assert repo._cached_qa is None
    
    def test_caching_behavior(self, repo_with_temp_paths):
        """Repository should cache data and reuse it"""
        repo, data_dir = repo_with_temp_paths
        
        # First access should populate cache
        tau1 = repo.get_tau()
        assert repo._cached_tau == tau1
        
        # Second access should use cache (same object)
        tau2 = repo.get_tau()
        assert tau1 == tau2
        
        # Similar for other methods
        ids1 = repo.get_image_ids()
        ids2 = repo.get_image_ids()
        assert ids1 == ids2
        
        qa1 = repo.get_all_qa()
        qa2 = repo.get_all_qa()
        assert qa1 == qa2


class TestFileRepositoryIntegration:
    """Integration tests for FileRepository with real file operations"""
    
    @pytest.fixture
    def temp_repo_dir(self):
        """Create a temporary directory with real FileRepository setup"""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "images").mkdir(parents=True, exist_ok=True)
        
        yield temp_dir, data_dir
        shutil.rmtree(temp_dir)
    
    def test_file_persistence(self, temp_repo_dir):
        """Test that data persists to actual files"""
        temp_dir, data_dir = temp_repo_dir
        
        with patch('app.file_repository.QA_PATH', data_dir / "qa.jsonl"), \
             patch('app.file_repository.IDS_PATH', data_dir / "ids.json"), \
             patch('app.file_repository.TAU_PATH', data_dir / "tau.txt"), \
             patch('app.file_repository.IMAGES_DIR', data_dir / "images"):
            
            repo = FileRepository()
            
            # Add data
            repo.set_tau(0.42)
            repo.add_image_id("persist_test.jpg")
            repo.upsert_qa("persist_test.jpg", "Persistent answer", 5)
            
            # Verify files exist
            assert (data_dir / "tau.txt").exists()
            assert (data_dir / "ids.json").exists()
            assert (data_dir / "qa.jsonl").exists()
            
            # Verify file contents
            tau_content = (data_dir / "tau.txt").read_text()
            assert float(tau_content.strip()) == 0.42
            
            ids_content = json.loads((data_dir / "ids.json").read_text())
            assert ids_content == ["persist_test.jpg"]
            
            qa_lines = (data_dir / "qa.jsonl").read_text().strip().split('\n')
            qa_record = json.loads(qa_lines[0])
            assert qa_record["id"] == "persist_test.jpg"
            assert qa_record["answer"] == "Persistent answer"
            assert qa_record["quality"] == 5