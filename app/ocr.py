"""
OCR text extraction module with support for PaddleOCR and Tesseract.
Provides unified interface with automatic fallback and confidence scoring.
"""
import os, logging, hashlib
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class OCRResult:
    """Container for OCR extraction results."""
    def __init__(self, text: str, confidence: float, language: str = "unknown", 
                 engine: str = "unknown", bboxes: List = None):
        self.text = text.strip()
        self.confidence = float(confidence)
        self.language = language
        self.engine = engine
        self.bboxes = bboxes or []
        self.hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "engine": self.engine,
            "hash": self.hash,
            "bbox_count": len(self.bboxes)
        }

class OCREngine:
    """Unified OCR interface with PaddleOCR and Tesseract support."""
    
    def __init__(self, prefer_gpu: bool = False, confidence_threshold: float = 0.5):
        self.prefer_gpu = prefer_gpu
        self.confidence_threshold = confidence_threshold
        self._paddleocr = None
        self._tesseract_available = self._check_tesseract()
        self._cache = {}
        
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning("Tesseract not available")
            return False
    
    def _get_paddleocr(self):
        """Lazy load PaddleOCR to avoid startup overhead."""
        if self._paddleocr is None:
            try:
                from paddleocr import PaddleOCR
                use_gpu = self.prefer_gpu
                # Initialize with multilingual support
                self._paddleocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',  # Primary language, can detect others
                    use_gpu=use_gpu,
                    show_log=False
                )
                logger.info(f"PaddleOCR initialized (GPU: {use_gpu})")
            except Exception as e:
                logger.warning(f"PaddleOCR not available: {e}")
                self._paddleocr = False
        return self._paddleocr if self._paddleocr is not False else None
    
    def _image_hash(self, image_path: Union[str, Path]) -> str:
        """Generate hash for image caching."""
        path = Path(image_path)
        stat = path.stat()
        content = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Preprocess image for better OCR results."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply basic enhancement
        # Increase contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Reduce noise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def _extract_with_paddleocr(self, image_path: Union[str, Path]) -> Optional[OCRResult]:
        """Extract text using PaddleOCR."""
        paddle = self._get_paddleocr()
        if not paddle:
            return None
        
        try:
            # PaddleOCR can handle path strings directly
            result = paddle.ocr(str(image_path), cls=True)
            
            if not result or not result[0]:
                return OCRResult("", 0.0, "unknown", "paddleocr")
            
            # Parse PaddleOCR output: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
            texts = []
            confidences = []
            bboxes = []
            
            for line in result[0]:
                if len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if confidence >= self.confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)
                        bboxes.append(bbox)
            
            if not texts:
                return OCRResult("", 0.0, "unknown", "paddleocr")
            
            combined_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                language="auto",  # PaddleOCR auto-detects
                engine="paddleocr",
                bboxes=bboxes
            )
            
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed for {image_path}: {e}")
            return None
    
    def _extract_with_tesseract(self, image_path: Union[str, Path]) -> Optional[OCRResult]:
        """Extract text using Tesseract."""
        if not self._tesseract_available:
            return None
        
        try:
            import pytesseract
            
            # Use preprocessed image for better results
            processed_img = self._preprocess_image(image_path)
            
            # Get text with confidence
            data = pytesseract.image_to_data(
                processed_img, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume a single uniform block of text
            )
            
            # Filter by confidence and combine text
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) >= (self.confidence_threshold * 100):
                    text = data['text'][i].strip()
                    if text:  # Skip empty text
                        texts.append(text)
                        confidences.append(int(data['conf'][i]) / 100.0)
            
            if not texts:
                return OCRResult("", 0.0, "unknown", "tesseract")
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            # Detect language
            try:
                lang_data = pytesseract.image_to_osd(processed_img)
                language = "unknown"
                for line in lang_data.split('\n'):
                    if 'Script:' in line:
                        language = line.split(':')[-1].strip()
                        break
            except:
                language = "unknown"
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                language=language,
                engine="tesseract"
            )
            
        except Exception as e:
            logger.warning(f"Tesseract extraction failed for {image_path}: {e}")
            return None
    
    def extract_text(self, image_path: Union[str, Path], use_cache: bool = True) -> OCRResult:
        """
        Extract text from image with automatic engine selection and fallback.
        
        Args:
            image_path: Path to image file
            use_cache: Whether to use cached results
            
        Returns:
            OCRResult with text, confidence, and metadata
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Check cache
        cache_key = self._image_hash(path)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try PaddleOCR first (higher accuracy)
        result = self._extract_with_paddleocr(path)
        
        # Fallback to Tesseract if PaddleOCR fails or gives poor results
        if (result is None or 
            (result.confidence < self.confidence_threshold and self._tesseract_available)):
            
            tesseract_result = self._extract_with_tesseract(path)
            if tesseract_result and tesseract_result.confidence > (result.confidence if result else 0):
                result = tesseract_result
        
        # Final fallback
        if result is None:
            result = OCRResult("", 0.0, "unknown", "none")
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
            
        return result
    
    def batch_extract(self, image_paths: List[Union[str, Path]], 
                     show_progress: bool = True) -> Dict[str, OCRResult]:
        """
        Extract text from multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping image paths to OCRResult objects
        """
        results = {}
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Extracting OCR text")
            except ImportError:
                iterator = image_paths
        else:
            iterator = image_paths
        
        for path in iterator:
            try:
                results[str(path)] = self.extract_text(path)
            except Exception as e:
                logger.error(f"Failed to extract text from {path}: {e}")
                results[str(path)] = OCRResult("", 0.0, "unknown", "error")
        
        return results
    
    def get_engine_status(self) -> Dict[str, bool]:
        """Get availability status of OCR engines."""
        return {
            "paddleocr": self._get_paddleocr() is not None,
            "tesseract": self._tesseract_available,
            "gpu_available": self.prefer_gpu
        }


# Global OCR engine instance
_ocr_engine = None

def get_ocr_engine(prefer_gpu: bool = False, confidence_threshold: float = 0.5) -> OCREngine:
    """Get or create global OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(prefer_gpu=prefer_gpu, confidence_threshold=confidence_threshold)
    return _ocr_engine

def extract_text_simple(image_path: Union[str, Path]) -> str:
    """Simple interface to extract text from a single image."""
    engine = get_ocr_engine()
    result = engine.extract_text(image_path)
    return result.text