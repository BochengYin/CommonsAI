# OCR Text Search Architecture

## Design Decisions

### OCR Library Selection
**Primary**: PaddleOCR (96% accuracy, best for complex layouts, GPU optimized)
**Fallback**: Tesseract (100+ languages, CPU optimized, reliable baseline)

**Rationale**: 
- PaddleOCR offers superior accuracy (96.58% vs ~85-90% for Tesseract)
- Better handling of complex layouts and multi-language documents
- GPU acceleration available when needed
- Tesseract as fallback ensures maximum language coverage

### Pipeline Architecture

```
Image Input
    ↓
OCR Engine Selection (based on availability/performance)
    ↓ 
Text Extraction (with confidence scores)
    ↓
Text Preprocessing (cleanup, normalization)
    ↓
Multi-Channel Encoding:
├── OpenCLIP Text Embedding (semantic)
├── BM25 Lexical Indexing (keyword)
└── Text Storage (raw + processed)
    ↓
Hybrid Retrieval via RRF Fusion
```

### Data Flow

**Index Building (`build_index.py`)**:
```
1. Load images from data/images/
2. Extract text via OCR pipeline
3. Generate embeddings (image + OCR text)
4. Build FAISS index (image vectors)
5. Build BM25 index (OCR text)
6. Store: img_embeds.npy, ocr_text.json, bm25.index
```

**Query Processing (`server.py`)**:
```
1. Receive text query
2. Parallel channel retrieval:
   - Text→Image (OpenCLIP)
   - Text→OCR Text (OpenCLIP) 
   - Text→OCR Text (BM25)
3. RRF fusion: score = Σ(1/(k+rank))
4. Return Top-K with confidence scores
```

### Storage Schema Extensions

**New Files**:
- `data/ocr_text.json`: {img_id: {text, confidence, language}}
- `data/bm25.index`: BM25 corpus and index
- `data/hybrid_embeds.npy`: Combined embeddings if needed

**Extended qa.jsonl**:
```json
{
  "id": "img_123.jpg",
  "answer": "...",
  "quality": 3,
  "path": "data/images/img_123.jpg",
  "ocr_text": "extracted text content",
  "ocr_confidence": 0.92,
  "ocr_language": "en",
  "fingerprints": {
    "text_hash": "sha256...",
    "sim_hash": "binary..."
  }
}
```

### Performance Optimization

**OCR Engine Selection Logic**:
```python
def select_ocr_engine(image_path, gpu_available=False):
    if gpu_available and paddleocr_available:
        return "paddleocr"
    elif tesseract_available:
        return "tesseract" 
    else:
        raise OCRNotAvailableError()
```

**Caching Strategy**:
- Cache OCR results by image hash to avoid re-processing
- Store confidence scores to enable quality-based fallbacks
- Lazy loading of OCR engines to minimize startup time

### RRF Fusion Formula

```python
def compute_rrf_score(ranks, k=60):
    """
    ranks: dict{"channel_name": rank} (1-indexed)
    k: RRF constant (typical: 60)
    """
    score = 0.0
    for channel, rank in ranks.items():
        if rank is not None:
            score += 1.0 / (k + rank)
    return score
```

### Error Handling & Fallbacks

1. **OCR Engine Failure**: PaddleOCR fails → Tesseract → Skip OCR channels
2. **Low Confidence Text**: confidence < 0.5 → exclude from BM25/embedding
3. **No Text Detected**: fallback to image-only retrieval
4. **GPU Unavailable**: automatically use CPU-optimized Tesseract

### Integration Points

**Embeddings Module** (`app/embeddings.py`):
- Add `encode_ocr_text()` function
- Support batch processing for efficiency
- Handle multilingual text encoding

**Server Module** (`app/server.py`):
- New endpoint: `/search_hybrid` 
- Extended `/query` with multi-channel results
- OCR status in `/health` endpoint

**Build Module** (`app/build_index.py`):
- Parallel OCR processing during index building
- Progress tracking for OCR extraction phase
- Error recovery and partial index building

### Success Metrics

**Target Performance**:
- Hit@3 accuracy: 20% → ≥35%
- p50 latency: 0.8s → <1.2s  
- OCR text recall: ≥80% for clear text images

**Quality Gates**:
- OCR confidence threshold: ≥0.5 for inclusion
- BM25 minimum doc frequency: ≥2 to reduce noise
- RRF weight tuning based on channel performance

This architecture enables cross-modal text-in-image search while maintaining the existing OpenCLIP image similarity foundation.