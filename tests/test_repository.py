import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from abc import ABC

from app.repository import Repository
from app.file_repository import FileRepository


class TestRepositoryInterface:
    """Test Repository abstract interface compliance"""
    
    def test_repository_is_abstract(self):
        """Repository should be abstract and not instantiable"""
        with pytest.raises(TypeError):
            Repository()
    
    def test_repository_has_required_methods(self):
        """Repository should define all required abstract methods"""
        required_methods = [
            'get_tau', 'set_tau', 'get_image_ids', 'add_image_id',
            'get_all_qa', 'get_qa_by_id', 'upsert_qa', 'refresh'
        ]
        
        for method_name in required_methods:
            assert hasattr(Repository, method_name)
            method = getattr(Repository, method_name)
            assert getattr(method, '__isabstractmethod__', False), f"{method_name} should be abstract"


class TestRepositoryContract:
    """Test that any Repository implementation follows the contract"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture  
    def mock_repository(self):
        """Create a mock repository that implements the interface"""
        repo = Mock(spec=Repository)
        repo.get_tau.return_value = 0.3
        repo.get_image_ids.return_value = ['img1.jpg', 'img2.png']
        repo.get_all_qa.return_value = [
            {'id': 'img1.jpg', 'answer': 'test answer', 'quality': 3},
            {'id': 'img2.png', 'answer': '', 'quality': 0}
        ]
        repo.get_qa_by_id.return_value = {'id': 'img1.jpg', 'answer': 'test answer', 'quality': 3}
        return repo
    
    def test_get_tau_returns_float(self, mock_repository):
        """get_tau should return a float"""
        tau = mock_repository.get_tau()
        assert isinstance(tau, float)
        assert 0.0 <= tau <= 1.0
    
    def test_get_image_ids_returns_list(self, mock_repository):
        """get_image_ids should return list of strings"""
        ids = mock_repository.get_image_ids()
        assert isinstance(ids, list)
        assert all(isinstance(img_id, str) for img_id in ids)
    
    def test_get_all_qa_returns_list_of_dicts(self, mock_repository):
        """get_all_qa should return list of dictionaries"""
        qa_records = mock_repository.get_all_qa()
        assert isinstance(qa_records, list)
        for record in qa_records:
            assert isinstance(record, dict)
            assert 'id' in record
            assert 'answer' in record  
            assert 'quality' in record
    
    def test_get_qa_by_id_contract(self, mock_repository):
        """get_qa_by_id should return dict or None"""
        result = mock_repository.get_qa_by_id('img1.jpg')
        assert result is None or isinstance(result, dict)
        if result:
            assert 'id' in result
            assert 'answer' in result
            assert 'quality' in result


class TestRepositoryEdgeCases:
    """Test Repository edge cases and error conditions"""
    
    @pytest.fixture
    def mock_repository(self):
        repo = Mock(spec=Repository)
        return repo
    
    def test_nonexistent_qa_id(self, mock_repository):
        """get_qa_by_id should handle nonexistent IDs gracefully"""
        mock_repository.get_qa_by_id.return_value = None
        result = mock_repository.get_qa_by_id('nonexistent.jpg')
        assert result is None
    
    def test_empty_repository(self, mock_repository):
        """Repository should handle empty state gracefully"""
        mock_repository.get_image_ids.return_value = []
        mock_repository.get_all_qa.return_value = []
        
        assert mock_repository.get_image_ids() == []
        assert mock_repository.get_all_qa() == []
    
    def test_tau_boundary_values(self, mock_repository):
        """tau should handle boundary values correctly"""
        test_values = [0.0, 0.5, 1.0]
        
        for tau_value in test_values:
            mock_repository.get_tau.return_value = tau_value
            result = mock_repository.get_tau()
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0