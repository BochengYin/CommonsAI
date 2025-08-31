# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CommonsAI is a community Q&A index system for shared LLM answers. It uses cross-modal retrieval to encode both text and images into the same semantic space using OpenCLIP, allowing text questions to find matching image threads and vice versa. The system includes a thread selection flow where users can either join existing threads or start new LLM-assisted threads.

## Architecture

### Core Components
- **OpenCLIP Integration**: Uses ViT-B-32 model with laion2b_s34b_b79k pretrained weights for text/image encoding
- **FAISS Vector Search**: IndexFlatIP for cosine similarity search on normalized embeddings
- **FastAPI Server**: REST API with endpoints for querying, updating answers, and managing images
- **Local Storage**: Uses .npy files for embeddings, FAISS index files, and JSONL for Q&A data

### Key Files
- `app/embeddings.py`: OpenCLIP model loading and encoding functions
- `app/server.py`: FastAPI application with all API endpoints
- `app/build_index.py`: Index building pipeline for image processing
- `app/utils.py`: File I/O utilities and path constants
- `scripts/rank.py`: CLI tool for ranking images by text similarity

### Data Structure
- `data/images/`: Image files (jpg/jpeg/png)
- `data/img_embeds.npy`: Encoded image vectors (N, 512)
- `data/ids.json`: Maps array indices to image filenames
- `data/img.index`: FAISS index file
- `data/qa.jsonl`: Question-answer data with quality scores
- `data/tau.txt`: Similarity threshold for HIT/MISS decisions

## Common Development Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Build Index
```bash
# Put images in data/images/ first
python -m app.build_index
```

### Run Development Server
```bash
uvicorn app.server:app --reload --port 8000
```

### CLI Query Tool
```bash
# Query with text to see ranked image matches
python scripts/rank.py --text "largest mammal whale" --k 5
python scripts/rank.py --text "shakespeare play with Romeo" --k 5
```

### API Endpoints
- `GET /health` - System status and image count
- `POST /query` - Search for similar images (form-data: text, k)
- `POST /update_answer` - Update Q&A data (form-data: img_id, answer, quality)
- `POST /add_image` - Add new image to index (form-data: file)
- `POST /set_tau` - Adjust similarity threshold (form-data: tau)

## Development Notes

### OpenCLIP Integration
- Model initialization is lazy-loaded for stability
- Uses validation transform (`preprocess_val`) for inference
- Text and image encodings are L2-normalized for cosine similarity
- Device detection: CUDA if available, otherwise CPU

### FAISS Import Ordering
On macOS/ARM systems, there can be ABI conflicts between PyTorch and FAISS. The codebase uses a specific import pattern:
1. Compute embeddings first (loads PyTorch/OpenCLIP)
2. Import FAISS after encoding is complete
This pattern is used in `build_index.py` and `scripts/rank.py`.

### Similarity Threshold (Tau)
The system uses a configurable threshold (`tau`) to decide between HIT (return cached answer) and MISS (needs new LLM query):
- HIT: `sim >= tau AND answer exists AND quality >= 3`
- MISS: Otherwise
- Default tau: 0.30 (stored in `data/tau.txt`)

### Quality Scoring
Community answers are scored 0-5 where:
- 0: No answer or placeholder
- 3+: Acceptable quality threshold for HIT decisions
- Higher scores indicate better community validation

## Testing and Validation

### Metrics Tracking
Current baseline (v0.1): ~20% Hit@3 accuracy, ~0.8s p50 latency
Target for v0.2: ≥35% Hit@3, <1.2s p50 latency with multi-modal retrieval

### Manual Testing
1. Build index with sample images
2. Test queries with `scripts/rank.py` 
3. Verify API endpoints with curl or HTTP client
4. Check similarity scores and HIT/MISS decisions

## File Organization

### Required Directories
- `data/images/`: Place historical images here before building index
- `data/`: Auto-created for embeddings, index, and metadata files
- `app/`: Core application modules
- `scripts/`: CLI utilities and tools

### Supported Formats
- Images: .jpg, .jpeg, .png only (other formats filtered out)
- Text encoding: UTF-8 for all JSON/JSONL files
- Dependencies: See requirements.txt for exact versions