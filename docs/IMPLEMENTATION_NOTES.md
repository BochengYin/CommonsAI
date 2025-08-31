# Implementation Notes

Technical debt, design decisions, and future improvements for CommonsAI v0.2 multi-modal hybrid retrieval.

## Current Implementation Status

### ✅ Completed (Phase 0)
- Documentation framework established
- CHANGELOG.md with semantic versioning
- Architecture Decision Records structure
- Migration guide template
- Enhanced research log tracking

### 🔄 In Progress
- *None currently*

### 📋 Planned
- Phase 1: OCR Pipeline + BM25 Lexical Search + Multi-Vector Storage
- Phase 2: RRF Fusion + Cross-Encoder Reranking  
- Phase 3: Full Testing + Schema Migration

## Design Decisions

### Documentation Strategy
- **Decision**: Multi-layered documentation approach (CHANGELOG + ADRs + Migration + Research Log)
- **Rationale**: Complex system changes need comprehensive tracking for maintainability
- **Trade-offs**: More overhead but better long-term understanding

## Technical Debt

### Current (v0.1)
1. **Single-modal limitation**: Only text→image matching via OpenCLIP
2. **No OCR extraction**: Question text embedded in images not utilized
3. **No lexical search**: Missing keyword/phrase matching capabilities
4. **Simple confidence scoring**: Basic threshold without multi-signal fusion
5. **Limited data schema**: qa.jsonl format lacks structured metadata

### Planned Improvements (v0.2)
1. **Multi-channel retrieval**: Text + Image + OCR + BM25 parallel search
2. **Advanced fusion**: RRF algorithm for combining multiple ranking signals
3. **Two-stage ranking**: Cross-encoder reranker for better precision
4. **Rich data model**: Full Trajectory/Recipe schema with versioning
5. **Comprehensive metrics**: Hit@K, latency, token savings tracking

## Performance Considerations

### Memory Usage
- **Current**: ~50MB for 6 images (single vector index)
- **Expected v0.2**: ~150-200MB (3 vector indices + BM25 index + reranker model)
- **Optimization**: Consider lazy loading, model quantization if needed

### Latency Profile
- **Current**: ~300ms for text encoding + FAISS search
- **Expected v0.2**: ~800ms for 4-channel search + reranking (still under 1.2s target)
- **Optimization**: Parallel channel processing, caching frequent queries

### Storage Growth
- **Current**: ~12KB per image (vector + metadata)
- **Expected v0.2**: ~36KB per image (3 vectors + OCR text + extended metadata)
- **Consideration**: Implement cleanup policies for old/low-quality entries

## Known Limitations

### OCR Accuracy
- **Issue**: OCR may struggle with stylized fonts, low resolution, handwritten text
- **Mitigation**: Multiple OCR engines, confidence scoring, manual correction flow

### BM25 Cold Start
- **Issue**: Lexical search requires sufficient text corpus to be effective
- **Mitigation**: Hybrid approach where vector search compensates for sparse text

### Cross-encoder Latency
- **Issue**: Reranking adds compute overhead, especially for large candidate sets
- **Mitigation**: Limit reranking to top-K candidates, use efficient models

## Future Improvements

### Phase 4+ (Post v0.2)
1. **Multi-lingual support**: Extend beyond English OCR and embeddings
2. **Active learning**: User feedback to improve ranking and OCR accuracy
3. **Semantic chunking**: Break long answers into reusable components
4. **Graph-based recipes**: Link related problems and solutions
5. **Real-time updates**: Incremental index updates without full rebuilds

### Advanced Features
1. **Visual similarity**: Image-to-image matching for screenshots/diagrams
2. **Code understanding**: Specialized embeddings for programming contexts
3. **Temporal relevance**: Weight recent solutions higher for evolving domains
4. **Domain adaptation**: Fine-tuned models for specific verticals

## Debugging & Monitoring

### Key Metrics to Track
- Hit@1, Hit@3, Hit@5 accuracy by channel
- Query latency breakdown (embedding, search, fusion, rerank)
- Memory usage and index sizes
- OCR extraction success rate and confidence
- False positive/negative rates for confidence thresholds

### Diagnostic Tools Needed
- Query debugging endpoint showing per-channel scores
- OCR confidence visualization
- RRF weight optimization tools
- A/B testing framework for algorithm changes

## Dependencies & Risks

### External Dependencies
- **OpenCLIP**: Core embedding model (stable)
- **FAISS**: Vector search engine (stable)
- **Tesseract/PaddleOCR**: OCR engines (moderate risk of version conflicts)
- **Cross-encoder models**: Hugging Face transformers (model availability risk)

### Risk Mitigation
- Pin exact dependency versions in requirements.txt
- Provide fallback options (e.g., Tesseract + PaddleOCR)
- Cache models locally to reduce download failures
- Comprehensive error handling and graceful degradation

## Lessons Learned

*To be updated during implementation*

### Phase 1 Lessons
- *TBD*

### Phase 2 Lessons  
- *TBD*

### Phase 3 Lessons
- *TBD*

---

**Last Updated**: 2025-08-31  
**Next Review**: After each phase completion