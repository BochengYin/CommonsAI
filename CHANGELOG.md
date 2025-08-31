# Changelog

All notable changes to CommonsAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-modal hybrid retrieval system (in progress)
- OCR pipeline for text extraction from images
- BM25 lexical search alongside vector search
- RRF (Reciprocal Rank Fusion) for multi-channel score combination
- Cross-encoder reranker for improved accuracy
- Comprehensive documentation framework

### Changed
- Enhanced query processing with 4-channel parallel search
- Extended data schema to support multi-vector storage
- Improved API with hybrid query endpoint

### Performance
- Target: Hit@3 accuracy improvement from ~20% to ≥35%
- Target: p50 latency < 1.2s for instant answers
- Target: Token savings ≥40% through better hit rates

## [0.1.0] - 2025-08-31

### Added
- Initial MVP with OpenCLIP + FAISS implementation
- Cross-modal text→image retrieval
- FastAPI server with basic endpoints (`/health`, `/query`, `/update_answer`, `/add_image`, `/set_tau`)
- CLI ranking script for testing
- Basic Q&A storage in JSONL format
- Confidence threshold system with tau parameter

### Infrastructure
- Docker-ready setup with requirements.txt
- Data organization in `data/` directory
- Image storage and embedding pipeline
- FAISS IndexFlatIP for vector search

### Documentation
- README with setup instructions
- Research log for tracking experiments
- PRD (Product Requirements Document) v0.1

### Known Limitations
- Single-modal matching (text query → image vectors only)
- No OCR text extraction
- No lexical/keyword search
- No hybrid retrieval or reranking
- Limited accuracy due to visual-only matching