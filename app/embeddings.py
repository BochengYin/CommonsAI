import torch, numpy as np
import open_clip
from PIL import Image
from typing import List, Union, Dict, Optional
from .ocr import OCRResult

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model: OpenCLIP ViT-B/32 (laion2b_s34b_b79k). Robust for multilingual text.
_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"

_model = None
_tokenizer = None
_preprocess = None

def _ensure_model():
    global _model, _tokenizer, _preprocess
    if _model is None:
        _model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED, device=_DEVICE
        )
        # Prefer the validation transform for inference
        _preprocess = preprocess_val
        _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        _model.eval()

@torch.no_grad()
def encode_text(texts):
    _ensure_model()
    if isinstance(texts, str):
        texts = [texts]
    toks = _tokenizer(texts).to(_DEVICE)
    feats = _model.encode_text(toks)
    feats = feats.float()
    feats /= feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu().numpy()

@torch.no_grad()
def encode_images(paths):
    _ensure_model()
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im)
    ims = [_preprocess(im).unsqueeze(0) for im in ims]
    batch = torch.cat(ims, dim=0).to(_DEVICE)
    feats = _model.encode_image(batch).float()
    feats /= feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu().numpy()

@torch.no_grad()
def encode_ocr_text(ocr_results: Union[OCRResult, List[OCRResult]]) -> np.ndarray:
    """
    Encode OCR extracted text using the same OpenCLIP model.
    
    Args:
        ocr_results: Single OCRResult or list of OCRResults
        
    Returns:
        Normalized embeddings array (N, embedding_dim)
    """
    if isinstance(ocr_results, OCRResult):
        ocr_results = [ocr_results]
    
    texts = []
    for result in ocr_results:
        # Only encode text with sufficient confidence and length
        if result.confidence >= 0.5 and len(result.text.strip()) > 0:
            texts.append(result.text)
        else:
            # Use empty string for low-confidence results (will get zero vector)
            texts.append("")
    
    if not texts:
        # Return zero vectors if no valid text
        _ensure_model()
        embedding_dim = _model.text_projection.out_features
        return np.zeros((len(ocr_results), embedding_dim), dtype=np.float32)
    
    return encode_text(texts)

@torch.no_grad() 
def encode_mixed_content(image_paths: List[str], ocr_results: List[OCRResult],
                        fusion_strategy: str = "concatenate") -> np.ndarray:
    """
    Encode both image and OCR text content with different fusion strategies.
    
    Args:
        image_paths: List of image file paths
        ocr_results: List of OCR results corresponding to images
        fusion_strategy: "concatenate", "average", or "weighted"
        
    Returns:
        Fused embeddings array (N, embedding_dim * fusion_factor)
    """
    if len(image_paths) != len(ocr_results):
        raise ValueError("Number of images and OCR results must match")
    
    img_embeds = encode_images(image_paths)
    ocr_embeds = encode_ocr_text(ocr_results)
    
    if fusion_strategy == "concatenate":
        # Simple concatenation: [img_embed, ocr_embed]
        return np.concatenate([img_embeds, ocr_embeds], axis=1)
        
    elif fusion_strategy == "average":
        # Element-wise average
        return (img_embeds + ocr_embeds) / 2.0
        
    elif fusion_strategy == "weighted":
        # Weight by OCR confidence
        weights = np.array([r.confidence for r in ocr_results]).reshape(-1, 1)
        # Normalize weights to [0.1, 0.9] range to avoid complete dominance
        weights = 0.1 + 0.8 * weights  
        return weights * ocr_embeds + (1 - weights) * img_embeds
        
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

def get_embedding_dim() -> int:
    """Get the embedding dimension of the current model."""
    _ensure_model()
    return _model.text_projection.out_features

def create_embedding_metadata(image_paths: List[str], 
                            ocr_results: List[OCRResult]) -> List[Dict]:
    """
    Create metadata for embeddings including OCR information.
    
    Args:
        image_paths: List of image file paths
        ocr_results: List of OCR results
        
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    for path, ocr_result in zip(image_paths, ocr_results):
        meta = {
            "image_path": str(path),
            "has_text": len(ocr_result.text) > 0,
            "ocr_confidence": ocr_result.confidence,
            "ocr_engine": ocr_result.engine,
            "text_length": len(ocr_result.text),
            "text_hash": ocr_result.hash,
            "language": ocr_result.language
        }
        metadata.append(meta)
    return metadata


