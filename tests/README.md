# CommonsAI Test Suite

Comprehensive testing for CommonsAI's Repository pattern and API endpoints.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Pytest configuration and shared fixtures
├── test_repository.py      # Repository interface tests
├── test_file_repository.py # FileRepository implementation tests
├── test_server.py          # FastAPI server integration tests
└── README.md              # This file
```

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# Repository layer tests
pytest -m repository

# Server/API tests
pytest -m server

# Exclude slow tests
pytest -m "not slow"
```

### Specific Test Files
```bash
pytest tests/test_repository.py
pytest tests/test_file_repository.py
pytest tests/test_server.py
```

### Specific Test Methods
```bash
pytest tests/test_repository.py::TestRepositoryInterface::test_repository_is_abstract
pytest tests/test_file_repository.py::TestFileRepository::test_upsert_qa_new_record
```

### Verbose Output
```bash
pytest -v -s
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Use mocks and fixtures to avoid external dependencies
- Fast execution

### Integration Tests (`@pytest.mark.integration`) 
- Test interaction between components
- May use temporary files and real Repository instances
- Slower execution

### Repository Tests (`@pytest.mark.repository`)
- Focus on Repository interface and FileRepository implementation
- Test data persistence, caching, and CRUD operations

### Server Tests (`@pytest.mark.server`)
- Test FastAPI endpoints and HTTP responses
- Use TestClient for API testing
- Test error handling and validation

## Key Fixtures

### `mock_repository`
Standard mock Repository for isolated testing:
```python
def test_something(mock_repository):
    mock_repository.get_tau.return_value = 0.5
    # Test logic here
```

### `isolated_temp_dir` 
Temporary directory for each test:
```python
def test_file_operations(isolated_temp_dir):
    test_file = isolated_temp_dir / "test.txt"
    test_file.write_text("test data")
```

### `test_data_structure`
Complete test data environment with files:
```python
def test_with_real_data(test_data_structure):
    data_dir = test_data_structure["data_dir"]
    qa_data = test_data_structure["sample_qa_data"]
    # Test with realistic data
```

### `sample_qa_data` & `sample_image_ids`
Realistic test data for Repository operations:
```python
def test_qa_operations(sample_qa_data):
    assert len(sample_qa_data) == 3
    assert sample_qa_data[0]["id"] == "whale.jpg"
```

## Test Coverage

### Repository Interface
- ✅ Abstract base class validation
- ✅ Method signature contracts
- ✅ Return type validation
- ✅ Edge case handling

### FileRepository Implementation  
- ✅ File I/O operations
- ✅ Caching behavior
- ✅ Data persistence
- ✅ CRUD operations (upsert, get, refresh)
- ✅ Error handling

### Server Integration
- ✅ All API endpoints (/health, /query, /set_tau, /update_answer)
- ✅ Request/response validation
- ✅ Repository integration
- ✅ Error handling and status codes
- ✅ Mock vs real Repository testing

## Writing New Tests

### Test Naming
- File: `test_<component>.py`
- Class: `Test<Component>` or `Test<Component><Aspect>`
- Method: `test_<behavior>_<condition>`

### Example Test Structure
```python
class TestNewComponent:
    """Test NewComponent functionality"""
    
    @pytest.fixture
    def component_setup(self):
        # Setup specific to this component
        yield setup_data
        # Cleanup
    
    def test_component_behavior(self, component_setup):
        """Test that component behaves correctly under normal conditions"""
        # Arrange
        component = NewComponent()
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result == expected_value
    
    def test_component_edge_case(self, component_setup):
        """Test component handles edge cases properly"""
        # Test implementation
        pass
```

### Best Practices
1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe the expected behavior
3. **Coverage**: Test happy path, edge cases, and error conditions
4. **Performance**: Keep unit tests fast, mark slow tests appropriately
5. **Fixtures**: Use appropriate fixtures for setup/teardown
6. **Mocking**: Mock external dependencies in unit tests

## Dependencies

Tests require these additional packages:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0  # For async test support
```

Install test dependencies:
```bash
pip install pytest pytest-asyncio
```