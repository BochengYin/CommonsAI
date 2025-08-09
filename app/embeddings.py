import torch, numpy as np
import open_clip
from PIL import Image

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


