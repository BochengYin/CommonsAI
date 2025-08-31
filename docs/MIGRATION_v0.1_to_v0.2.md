# Migration Guide: v0.1 to v0.2

This guide helps you upgrade from CommonsAI v0.1 (single-modal) to v0.2 (multi-modal hybrid retrieval).

## Overview

v0.2 introduces breaking changes to support multi-modal hybrid retrieval with OCR, BM25 lexical search, RRF fusion, and cross-encoder reranking.

## Breaking Changes

### Data Schema Changes

#### qa.jsonl Format
**v0.1**:
```json
{"id":"question1.png","type":"image","path":"data/images/question1.png","answer":"...","quality":5,"tags":["tag1"]}
```

**v0.2**:
```json
{
  "id": "question1.png",
  "type": "image", 
  "path": "data/images/question1.png",
  "question": "What is the largest mammal?",
  "answer": "The blue whale...",
  "quality": 5,
  "tags": ["biology", "mammals"],
  "ocr_text": "What is the largest mammal?",
  "embeddings": {
    "text": "text_embed_id",
    "image": "img_embed_id", 
    "ocr": "ocr_embed_id"
  },
  "created_at": "2025-08-31T00:00:00Z",
  "updated_at": "2025-08-31T00:00:00Z"
}
```

#### New Index Files
**v0.1**:
- `data/img.index` - single FAISS index
- `data/img_embeds.npy` - image embeddings
- `data/ids.json` - image ID mapping

**v0.2**:
- `data/text.index` - text embeddings FAISS index
- `data/img.index` - image embeddings FAISS index  
- `data/ocr.index` - OCR text embeddings FAISS index
- `data/lexical.index` - BM25 inverted index
- `data/embeddings/` - structured embedding storage
- `data/ids.json` - preserved for compatibility

### API Changes

#### New Endpoints
- `POST /hybrid_query` - Multi-modal hybrid search (recommended)
- `GET /rerank` - Cross-encoder reranking service

#### Modified Endpoints
- `POST /query` - Enhanced with multi-modal support (backward compatible)

#### New Query Parameters
- `channels` - Specify search channels: text,image,ocr,lexical (default: all)
- `rerank` - Enable/disable cross-encoder reranking (default: true)
- `rrf_k` - RRF constant for score fusion (default: 60)

## Migration Steps

### Step 1: Backup Current Data
```bash
cp -r data/ data_backup_v0.1/
```

### Step 2: Install New Dependencies
```bash
pip install -r requirements.txt  # Updated with OCR and reranking dependencies
```

### Step 3: Run Migration Script
```bash
python scripts/migrate_v0.1_to_v0.2.py
```

This script will:
1. Extract OCR text from existing images
2. Generate text embeddings for questions
3. Create new multi-vector indices
4. Update qa.jsonl schema
5. Preserve all existing data

### Step 4: Rebuild Indices
```bash
python -m app.build_index --multi-modal
```

### Step 5: Test Migration
```bash
# Test old endpoint (should still work)
curl -X POST "http://localhost:8000/query" -F "text=whale" -F "k=3"

# Test new hybrid endpoint
curl -X POST "http://localhost:8000/hybrid_query" -F "text=whale" -F "k=3" -F "channels=text,image,ocr,lexical"
```

## Configuration Changes

### Environment Variables
```bash
# Optional: Configure OCR engine
export OCR_ENGINE=tesseract  # or paddleocr

# Optional: Configure reranker model
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Optional: RRF fusion weights
export RRF_WEIGHTS_TEXT=0.4
export RRF_WEIGHTS_IMAGE=0.3  
export RRF_WEIGHTS_OCR=0.2
export RRF_WEIGHTS_LEXICAL=0.1
```

## Rollback Plan

If you need to rollback to v0.1:

1. Restore data backup:
   ```bash
   rm -rf data/
   mv data_backup_v0.1/ data/
   ```

2. Downgrade dependencies:
   ```bash
   git checkout v0.1.0
   pip install -r requirements.txt
   ```

3. Restart server:
   ```bash
   uvicorn app.server:app --reload --port 8000
   ```

## Performance Expectations

- **Accuracy**: Hit@3 should improve from ~20% to ≥35%
- **Latency**: First query may be slower due to multi-channel processing
- **Storage**: Expect 3-4x increase in index file sizes
- **Memory**: Higher memory usage due to multiple models

## Troubleshooting

### Common Issues

**OCR extraction fails**:
```bash
# Install system dependencies
brew install tesseract  # macOS
apt-get install tesseract-ocr  # Ubuntu
```

**Out of memory during indexing**:
```bash
# Reduce batch size
python -m app.build_index --batch-size 10
```

**Slow query performance**:
```bash
# Disable reranking temporarily
curl -X POST "/hybrid_query" -F "rerank=false" ...
```

## Support

For migration issues:
1. Check logs in `data/migration.log`
2. Review [Implementation Notes](IMPLEMENTATION_NOTES.md)
3. Open issue on GitHub with migration logs