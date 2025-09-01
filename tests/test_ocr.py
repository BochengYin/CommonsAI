"""
Tests for OCR functionality including text extraction and hybrid search.
"""
import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch, MagicMock

from app.ocr import OCREngine, OCRResult, get_ocr_engine, extract_text_simple
from app.embeddings import encode_ocr_text, encode_text
from app.bm25_search import BM25SearchEngine
from app.hybrid_retrieval import HybridRetriever, SearchResult

@pytest.fixture
def sample_text_image():
    """Create a sample image with text for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Create image with text
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text = "Hello World Test 123"
        draw.text((10, 30), text, fill='black', font=font)
        
        img.save(tmp.name)
        yield tmp.name, text
        
        # Cleanup
        Path(tmp.name).unlink()

@pytest.fixture
def mock_paddleocr():
    """Mock PaddleOCR for testing."""
    with patch('app.ocr.PaddleOCR') as mock_paddle_class:
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [
            [
                [[[10, 10], [100, 10], [100, 40], [10, 40]], ("Hello World", 0.95)],
                [[[10, 50], [80, 50], [80, 80], [10, 80]], ("Test 123", 0.88)]
            ]
        ]
        mock_paddle_class.return_value = mock_paddle
        yield mock_paddle

@pytest.fixture 
def mock_tesseract():
    """Mock Tesseract for testing."""
    with patch('app.ocr.pytesseract') as mock_tess:
        mock_tess.get_tesseract_version.return_value = "5.0.0"
        mock_tess.image_to_data.return_value = {
            'text': ['', 'Hello', 'World', 'Test', '123'],
            'conf': [0, 95, 92, 88, 90]
        }
        mock_tess.image_to_osd.return_value = "Script: Latin"
        yield mock_tess

class TestOCRResult:
    """Test OCRResult container class."""
    
    def test_create_ocr_result(self):
        result = OCRResult("Hello World", 0.95, "en", "tesseract")
        
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.engine == "tesseract"
        assert len(result.hash) == 16  # SHA256 truncated
    
    def test_ocr_result_to_dict(self):
        result = OCRResult("Test text", 0.88, "en", "paddleocr", [[[0,0],[100,50]]])
        
        result_dict = result.to_dict()
        expected_keys = {"text", "confidence", "language", "engine", "hash", "bbox_count"}
        
        assert set(result_dict.keys()) == expected_keys
        assert result_dict["bbox_count"] == 1
        assert result_dict["text"] == "Test text"

class TestOCREngine:
    """Test OCR engine functionality."""
    
    def test_init_ocr_engine(self):
        engine = OCREngine(prefer_gpu=False, confidence_threshold=0.6)
        
        assert engine.prefer_gpu == False
        assert engine.confidence_threshold == 0.6
        assert engine._paddleocr is None
        assert engine._cache == {}
    
    def test_check_tesseract_available(self):
        engine = OCREngine()
        # This will depend on system availability
        status = engine._check_tesseract()
        assert isinstance(status, bool)
    
    @patch('app.ocr.cv2')
    def test_preprocess_image(self, mock_cv2, sample_text_image):
        image_path, _ = sample_text_image
        
        # Mock cv2.imread to return a dummy image
        mock_cv2.imread.return_value = np.ones((100, 300, 3), dtype=np.uint8) * 255
        mock_cv2.cvtColor.return_value = np.ones((100, 300), dtype=np.uint8) * 255
        mock_cv2.convertScaleAbs.return_value = np.ones((100, 300), dtype=np.uint8) * 255
        mock_cv2.bilateralFilter.return_value = np.ones((100, 300), dtype=np.uint8) * 255
        
        engine = OCREngine()
        processed = engine._preprocess_image(image_path)
        
        assert processed is not None
        mock_cv2.imread.assert_called_once()
    
    def test_extract_with_paddleocr(self, mock_paddleocr, sample_text_image):
        image_path, expected_text = sample_text_image
        
        engine = OCREngine()
        result = engine._extract_with_paddleocr(image_path)
        
        assert result is not None
        assert isinstance(result, OCRResult)
        assert result.engine == "paddleocr"
        assert "Hello World" in result.text
        assert result.confidence > 0.5
    
    def test_extract_with_tesseract(self, mock_tesseract, sample_text_image):
        image_path, expected_text = sample_text_image
        
        with patch('app.ocr.cv2') as mock_cv2:
            # Mock image preprocessing 
            mock_cv2.imread.return_value = np.ones((100, 300, 3), dtype=np.uint8) * 255
            mock_cv2.cvtColor.return_value = np.ones((100, 300), dtype=np.uint8) * 255
            mock_cv2.convertScaleAbs.return_value = np.ones((100, 300), dtype=np.uint8) * 255
            mock_cv2.bilateralFilter.return_value = np.ones((100, 300), dtype=np.uint8) * 255
            
            engine = OCREngine()
            result = engine._extract_with_tesseract(image_path)
            
            assert result is not None
            assert isinstance(result, OCRResult)
            assert result.engine == "tesseract"
            assert "Hello World" in result.text

    def test_extract_text_with_cache(self, mock_paddleocr, sample_text_image):
        image_path, _ = sample_text_image
        
        engine = OCREngine()
        
        # First call should extract and cache
        result1 = engine.extract_text(image_path, use_cache=True)
        assert len(engine._cache) == 1
        
        # Second call should use cache
        result2 = engine.extract_text(image_path, use_cache=True)
        assert result1.text == result2.text
        assert result1.hash == result2.hash
    
    def test_batch_extract(self, mock_paddleocr, sample_text_image):
        image_path, _ = sample_text_image
        
        engine = OCREngine()
        results = engine.batch_extract([image_path], show_progress=False)
        
        assert len(results) == 1
        assert image_path in results
        assert isinstance(results[image_path], OCRResult)
    
    def test_get_engine_status(self):
        engine = OCREngine()
        status = engine.get_engine_status()
        
        expected_keys = {"paddleocr", "tesseract", "gpu_available"}
        assert set(status.keys()) == expected_keys
        assert isinstance(status["gpu_available"], bool)

class TestEmbeddings:
    """Test OCR text embedding functionality."""
    
    @patch('app.embeddings.encode_text')
    def test_encode_ocr_text_single(self, mock_encode_text):
        mock_encode_text.return_value = np.array([[0.1, 0.2, 0.3]])
        
        ocr_result = OCRResult("Test text", 0.95, "en", "tesseract")
        embeddings = encode_ocr_text(ocr_result)
        
        assert embeddings.shape == (1, 3)
        mock_encode_text.assert_called_once_with(["Test text"])
    
    @patch('app.embeddings.encode_text')
    def test_encode_ocr_text_batch(self, mock_encode_text):
        mock_encode_text.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        ocr_results = [
            OCRResult("First text", 0.9, "en", "tesseract"),
            OCRResult("Second text", 0.8, "en", "paddleocr")
        ]
        
        embeddings = encode_ocr_text(ocr_results)
        
        assert embeddings.shape == (2, 3)
        mock_encode_text.assert_called_once_with(["First text", "Second text"])
    
    @patch('app.embeddings.encode_text')
    def test_encode_ocr_text_low_confidence(self, mock_encode_text):
        mock_encode_text.return_value = np.array([[0.0, 0.0, 0.0]])
        
        # Low confidence result should be filtered out
        ocr_result = OCRResult("Low confidence", 0.3, "en", "tesseract")
        embeddings = encode_ocr_text(ocr_result)
        
        # Should encode empty string for low confidence
        mock_encode_text.assert_called_once_with([""])

class TestBM25Search:
    """Test BM25 lexical search functionality."""
    
    def test_bm25_search_engine_init(self):
        engine = BM25SearchEngine()
        
        assert engine.bm25 is None
        assert engine._loaded == False
    
    def test_bm25_search_no_index(self):
        engine = BM25SearchEngine(Path("/nonexistent/path"))
        
        success = engine.load_index()
        assert success == False
    
    @patch('app.bm25_search.pickle')
    def test_bm25_search_with_results(self, mock_pickle):
        from rank_bm25 import BM25Okapi
        
        # Mock BM25 data
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([0.5, 1.2, 0.8])
        mock_bm25.corpus = [["hello", "world"], ["test", "document"], ["example", "text"]]
        
        mock_data = {
            "bm25": mock_bm25,
            "image_indices": [0, 1, 2],
            "image_ids": ["img1.jpg", "img2.jpg", "img3.jpg"]
        }
        
        mock_pickle.load.return_value = mock_data
        
        with patch('pathlib.Path.exists', return_value=True):
            engine = BM25SearchEngine()
            engine.load_index()
            
            results = engine.search("test query", k=3)
            
            assert len(results) <= 3
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
            # Results should be sorted by score descending
            if len(results) > 1:
                assert results[0][1] >= results[1][1]

class TestHybridRetrieval:
    """Test hybrid retrieval with RRF fusion."""
    
    @patch('app.hybrid_retrieval.read_json')
    @patch('app.hybrid_retrieval.faiss.read_index')
    @patch('app.hybrid_retrieval.np.load')
    def test_hybrid_retriever_init(self, mock_np_load, mock_faiss_read, mock_read_json):
        # Mock successful loading
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_faiss_read.return_value = mock_index
        
        mock_np_load.return_value = np.random.rand(3, 512)
        mock_read_json.return_value = ["img1.jpg", "img2.jpg", "img3.jpg"]
        
        with patch('pathlib.Path.exists', return_value=True):
            retriever = HybridRetriever(rrf_k=60)
            success = retriever.load_indices()
            
            assert success == True
            assert retriever._loaded == True
            assert retriever.rrf_k == 60
    
    def test_compute_rrf_scores(self):
        retriever = HybridRetriever(rrf_k=60)
        
        # Mock channel results
        channel_results = {
            "channel1": [
                SearchResult("img1", 0.9, 1, "channel1"),
                SearchResult("img2", 0.8, 2, "channel1")
            ],
            "channel2": [
                SearchResult("img2", 0.95, 1, "channel2"), 
                SearchResult("img3", 0.7, 2, "channel2")
            ]
        }
        
        hybrid_results = retriever._compute_rrf_scores(channel_results)
        
        # Should have results for all unique images
        assert len(hybrid_results) == 3
        
        # Results should be sorted by RRF score
        assert hybrid_results[0].rrf_score >= hybrid_results[1].rrf_score
        
        # img2 should have highest score (appears in both channels with good ranks)
        img2_result = next(r for r in hybrid_results if r.image_id == "img2")
        assert len(img2_result.channels_matched) == 2
        assert img2_result.rrf_score > 0
    
    def test_search_result_dataclass(self):
        result = SearchResult("img1", 0.85, 1, "test_channel", {"meta": "data"})
        
        assert result.image_id == "img1"
        assert result.score == 0.85
        assert result.rank == 1
        assert result.channel == "test_channel"
        assert result.metadata == {"meta": "data"}

class TestIntegration:
    """Integration tests for OCR pipeline."""
    
    def test_simple_extraction_interface(self, sample_text_image):
        image_path, expected_text = sample_text_image
        
        with patch('app.ocr.get_ocr_engine') as mock_get_engine:
            mock_engine = MagicMock()
            mock_result = OCRResult("Hello World Test 123", 0.9, "en", "mock")
            mock_engine.extract_text.return_value = mock_result
            mock_get_engine.return_value = mock_engine
            
            result_text = extract_text_simple(image_path)
            assert result_text == "Hello World Test 123"
    
    @patch('app.embeddings._ensure_model')
    @patch('app.embeddings._model')
    def test_ocr_embedding_pipeline(self, mock_model, mock_ensure):
        """Test the complete OCR -> embedding pipeline."""
        # Mock OpenCLIP model
        mock_model.encode_text.return_value = MagicMock()
        mock_model.encode_text.return_value.float.return_value.cpu.return_value.numpy.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Create OCR result
        ocr_result = OCRResult("Sample text", 0.9, "en", "tesseract")
        
        # Test embedding generation
        embeddings = encode_ocr_text(ocr_result)
        
        assert embeddings is not None
        assert embeddings.shape[1] == 3  # Embedding dimension
        
    def test_global_ocr_engine(self):
        """Test global OCR engine singleton."""
        engine1 = get_ocr_engine()
        engine2 = get_ocr_engine()
        
        # Should return the same instance
        assert engine1 is engine2