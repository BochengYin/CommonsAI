# Data Schemas - OCR Extensions

## Overview

CommonsAI v0.2 extends the original image-only schemas to support multi-modal OCR text search. This document describes the data structures and file formats for the enhanced system.

## File Structure

```
data/
├── images/              # Original image files
├── img_embeds.npy      # Image embeddings (OpenCLIP visual)
├── ocr_embeds.npy      # OCR text embeddings (OpenCLIP text) 
├── ocr_text.json       # Raw OCR extraction results
├── img.index           # FAISS image similarity index
├── bm25.pkl           # BM25 lexical search index
├── ids.json           # Image ID mappings
├── qa.jsonl           # Q&A records with OCR metadata
└── tau.txt            # Similarity threshold
```

## Schema Definitions

### 1. OCR Text Storage (`ocr_text.json`)

Maps image IDs to OCR extraction results:

```json
{
  "img_001.jpg": {
    "text": "Machine Learning Course\nChapter 5: Neural Networks",
    "confidence": 0.94,
    "language": "en",
    "engine": "paddleocr",
    "hash": "a1b2c3d4e5f6789a",
    "bbox_count": 12
  },
  "img_002.png": {
    "text": "",
    "confidence": 0.0,
    "language": "unknown",
    "engine": "error",
    "hash": "0000000000000000",
    "bbox_count": 0
  }
}
```

**Field Descriptions:**
- `text`: Extracted text content (empty if no text or low confidence)
- `confidence`: OCR confidence score [0.0-1.0]
- `language`: Detected language code or "unknown"
- `engine`: OCR engine used ("paddleocr", "tesseract", "error")
- `hash`: SHA256 hash of text content (first 16 chars)
- `bbox_count`: Number of text bounding boxes detected

### 2. Extended Q&A Records (`qa.jsonl`)

Each line contains a JSON record with OCR metadata:

```jsonl
{"id": "img_001.jpg", "type": "image", "path": "data/images/img_001.jpg", "answer": "Neural networks are...", "quality": 4, "tags": ["ml", "education"], "ocr_text": "Machine Learning Course\nChapter 5: Neural Networks", "ocr_confidence": 0.94, "ocr_language": "en", "ocr_engine": "paddleocr", "text_hash": "a1b2c3d4e5f6789a"}
{"id": "img_002.png", "type": "image", "path": "data/images/img_002.png", "answer": "", "quality": 0, "tags": [], "ocr_text": "", "ocr_confidence": 0.0, "ocr_language": "unknown", "ocr_engine": "tesseract", "text_hash": "0000000000000000"}
```

**New OCR Fields:**
- `ocr_text`: Full extracted text content
- `ocr_confidence`: Overall confidence score
- `ocr_language`: Primary detected language
- `ocr_engine`: Engine that produced the result
- `text_hash`: Content fingerprint for deduplication

### 3. BM25 Index Structure (`bm25.pkl`)

Pickled dictionary containing BM25 search components:

```python
{
    "bm25": BM25Okapi,                    # Trained BM25 model
    "image_indices": [0, 2, 5, 8],       # Maps BM25 doc index to image index
    "image_ids": ["img_001.jpg", "img_003.jpg", ...]  # Corresponding image IDs
}
```

**Usage Pattern:**
1. Tokenize query → `bm25.get_scores(tokens)`
2. Map scores to image IDs via `image_ids[i]`
3. Return top-k results with scores

### 4. Embedding Arrays

#### Image Embeddings (`img_embeds.npy`)
- **Shape**: `(N, 512)` where N = number of images
- **Type**: `float32` normalized vectors
- **Source**: OpenCLIP ViT-B/32 image encoder
- **Index**: Maps to `ids.json` positions

#### OCR Text Embeddings (`ocr_embeds.npy`)
- **Shape**: `(N, 512)` matching image count
- **Type**: `float32` normalized vectors  
- **Source**: OpenCLIP ViT-B/32 text encoder applied to OCR text
- **Index**: Maps to same positions as image embeddings
- **Note**: Zero vectors for images without text (confidence < 0.5)

### 5. Image ID Mapping (`ids.json`)

Array mapping array indices to image filenames:

```json
["img_001.jpg", "img_002.png", "img_003.jpg", ...]
```

**Critical Invariant**: All arrays (image_embeds, ocr_embeds, FAISS index) must have the same length and ordering as this ID list.

## API Response Schemas

### Hybrid Search Response (`/search_hybrid`)

```json
{
  "decision": "HIT",
  "tau": 0.30,
  "search_type": "hybrid",
  "channels_used": ["image_similarity", "ocr_similarity", "bm25_lexical"],
  "topk": [
    {
      "img_id": "img_001.jpg",
      "rrf_score": 0.0847,
      "final_rank": 1,
      "channels_matched": ["ocr_similarity", "bm25_lexical"],
      "individual_scores": {
        "image_similarity": 0.0,
        "ocr_similarity": 0.82,
        "bm25_lexical": 2.14
      },
      "answer": "Neural networks are computational models...",
      "quality": 4,
      "path": "data/images/img_001.jpg",
      "ocr_text": "Machine Learning Course\nChapter 5: Neural Networks",
      "ocr_confidence": 0.94
    }
  ]
}
```

### BM25 Search Response (`/search_bm25`)

```json
{
  "search_type": "bm25_lexical",
  "total_matches": 3,
  "results": [
    {
      "img_id": "img_001.jpg",
      "bm25_score": 2.14,
      "matched_tokens": ["machine", "learning"],
      "match_count": 2,
      "answer": "...",
      "quality": 4,
      "path": "data/images/img_001.jpg",
      "ocr_text": "Machine Learning Course..."
    }
  ]
}
```

### OCR Status Response (`/ocr_status`)

```json
{
  "engine_availability": {
    "paddleocr": true,
    "tesseract": true,
    "gpu_available": false
  },
  "corpus_stats": {
    "total_images": 150,
    "images_with_ocr": 89,
    "avg_ocr_confidence": 0.78,
    "languages": ["en", "fr", "unknown"],
    "engines_used": ["paddleocr", "tesseract"]
  }
}
```

## Migration Guide

### From v0.1 to v0.2

**Automatic Migration**: Running `python -m app.build_index` on existing v0.1 data will:

1. **Preserve existing data**: `img_embeds.npy`, `img.index`, `ids.json` remain unchanged
2. **Add OCR processing**: Extract text from all images in `data/images/`
3. **Generate new files**: Create `ocr_embeds.npy`, `ocr_text.json`, `bm25.pkl`
4. **Extend qa.jsonl**: Add OCR fields to existing records, preserve existing answers
5. **Maintain compatibility**: v0.1 endpoints continue to work unchanged

**Manual Migration Steps** (if needed):
```bash
# 1. Backup existing data
cp -r data/ data_backup/

# 2. Install OCR dependencies
pip install paddleocr pytesseract opencv-python rank-bm25

# 3. Rebuild index with OCR processing
python -m app.build_index

# 4. Verify migration
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search_hybrid -d "text=your query"
```

### Data Validation

**Integrity Checks**:
```python
# Check array dimensions match
assert len(ids) == img_embeds.shape[0] == ocr_embeds.shape[0] == faiss_index.ntotal

# Check OCR coverage
ocr_records = [r for r in qa_records if r.get('ocr_text', '').strip()]
coverage = len(ocr_records) / len(qa_records)
print(f"OCR coverage: {coverage:.1%}")

# Check embedding consistency
assert np.allclose(np.linalg.norm(img_embeds, axis=1), 1.0)  # L2 normalized
assert np.allclose(np.linalg.norm(ocr_embeds, axis=1), 1.0, atol=1e-6)  # Allow zeros
```

## Performance Characteristics

### Storage Requirements
- **OCR Text**: ~2-50 KB per image (varies by text content)
- **OCR Embeddings**: 2 KB per image (512 × 4 bytes)
- **BM25 Index**: ~10-100 KB (depends on vocabulary size)
- **Total Overhead**: ~5-150 KB per image for OCR features

### Search Performance
- **Image-only**: ~10ms for 1000 images
- **OCR similarity**: ~15ms for 1000 images
- **BM25 search**: ~5ms for 1000 documents
- **Hybrid RRF fusion**: ~25ms total for 3 channels

### Index Building Time
- **OCR Extraction**: 2-10 seconds per image (GPU vs CPU)
- **Embedding Generation**: 50-200ms per image
- **BM25 Index**: <1 second for 1000 documents
- **Total**: Dominated by OCR extraction time

## Best Practices

### OCR Quality
- **Confidence Threshold**: Use ≥0.5 for reliable text
- **Image Quality**: Higher DPI images improve OCR accuracy  
- **Text Size**: Minimum 12pt font for reliable extraction
- **Language**: Specify expected language for better results

### Search Optimization
- **Channel Selection**: Use all channels for max recall
- **RRF Tuning**: k=60 works well, adjust based on data
- **Caching**: OCR results are cached by image hash
- **Batch Processing**: Process images in batches during indexing

### Data Management
- **Backup Strategy**: Include `ocr_text.json` in backups
- **Version Control**: Track schema versions in qa.jsonl
- **Monitoring**: Watch OCR confidence distributions
- **Cleanup**: Remove failed OCR results periodically