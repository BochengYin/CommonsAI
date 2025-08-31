"""
Pytest configuration and shared fixtures for CommonsAI tests
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from app.file_repository import FileRepository


@pytest.fixture(scope="session")
def temp_test_dir():
    """Session-scoped temporary directory for all tests"""
    temp_dir = tempfile.mkdtemp(prefix="commonsai_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def isolated_temp_dir():
    """Function-scoped temporary directory for individual tests"""
    temp_dir = tempfile.mkdtemp(prefix="commonsai_isolated_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_repository():
    """Standard mock repository for testing"""
    repo = Mock(spec=FileRepository)
    repo.get_tau.return_value = 0.3
    repo.get_image_ids.return_value = ["img1.jpg", "img2.png", "img3.jpg"]
    repo.get_all_qa.return_value = [
        {"id": "img1.jpg", "answer": "Answer 1", "quality": 4, "type": "image", "path": "data/images/img1.jpg", "tags": ["test"]},
        {"id": "img2.png", "answer": "", "quality": 0, "type": "image", "path": "data/images/img2.png", "tags": []},
        {"id": "img3.jpg", "answer": "Answer 3", "quality": 2, "type": "image", "path": "data/images/img3.jpg", "tags": ["low-quality"]}
    ]
    repo.get_qa_by_id.side_effect = lambda img_id: next(
        (record for record in repo.get_all_qa.return_value if record["id"] == img_id), 
        None
    )
    return repo


@pytest.fixture
def sample_qa_data():
    """Sample Q&A data for testing"""
    return [
        {
            "id": "whale.jpg",
            "type": "image", 
            "path": "data/images/whale.jpg",
            "answer": "Blue whales are the largest mammals on Earth",
            "quality": 5,
            "tags": ["mammal", "ocean", "largest"]
        },
        {
            "id": "python.png",
            "type": "image",
            "path": "data/images/python.png", 
            "answer": "Python is a high-level programming language",
            "quality": 4,
            "tags": ["programming", "language"]
        },
        {
            "id": "empty.jpg",
            "type": "image",
            "path": "data/images/empty.jpg",
            "answer": "",
            "quality": 0,
            "tags": []
        }
    ]


@pytest.fixture
def sample_image_ids():
    """Sample image IDs for testing"""
    return ["whale.jpg", "python.png", "empty.jpg", "test1.jpg", "test2.png"]


@pytest.fixture
def test_data_structure(isolated_temp_dir, sample_qa_data, sample_image_ids):
    """Create complete test data structure with files"""
    data_dir = isolated_temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images" 
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data files
    tau_path = data_dir / "tau.txt"
    ids_path = data_dir / "ids.json"
    qa_path = data_dir / "qa.jsonl"
    
    # Write tau
    tau_path.write_text("0.35")
    
    # Write IDs
    import json
    ids_path.write_text(json.dumps(sample_image_ids, indent=2))
    
    # Write QA data
    import orjson
    with qa_path.open("wb") as f:
        for record in sample_qa_data:
            f.write(orjson.dumps(record) + b"\n")
    
    return {
        "data_dir": data_dir,
        "images_dir": images_dir,
        "tau_path": tau_path,
        "ids_path": ids_path, 
        "qa_path": qa_path,
        "sample_qa_data": sample_qa_data,
        "sample_image_ids": sample_image_ids
    }


# Test categories for pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "repository: Tests for repository layer"
    )
    config.addinivalue_line(
        "markers", "server: Tests for server/API layer"
    )


# Pytest collection customization
def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on file/class names"""
    for item in items:
        # Mark based on test file name
        if "test_repository" in item.fspath.basename:
            item.add_marker(pytest.mark.repository)
        elif "test_server" in item.fspath.basename:
            item.add_marker(pytest.mark.server)
        
        # Mark based on test class name
        if hasattr(item, 'cls') and item.cls:
            if "Integration" in item.cls.__name__:
                item.add_marker(pytest.mark.integration)
            elif "Unit" in item.cls.__name__ or "Test" in item.cls.__name__:
                item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.name.lower() or "integration" in item.name.lower():
            item.add_marker(pytest.mark.slow)