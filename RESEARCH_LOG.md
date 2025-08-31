# Research Timeline

Keep short dated entries: context → hypothesis → method → metrics → findings → next.

## 2025-08-31
- Context: v0.2 Multi-modal implementation starting. Current system: single OpenCLIP, ~20% Hit@3 accuracy.
- Hypothesis: Multi-channel hybrid retrieval (text+image+OCR+BM25) + RRF + reranking will achieve ≥35% Hit@3.
- Method: Implement OCR → BM25 → RRF fusion → cross-encoder reranking in phases.
- Metrics: Target Hit@3 ≥35%, p50 latency <1.2s, token savings ≥40%.
- Findings: Documentation framework established. Ready for Phase 1 implementation.
- Next: Implement OCR pipeline with Tesseract/PaddleOCR comparison.

## 2025-08-09
- Context: MVP ready (OpenCLIP ViT-B/32; FAISS IP; FastAPI). Small demo image set.
- Hypothesis: Text→image retrieval works but needs semantically aligned images to shine.
- Method: Build index; query whale question; inspect similarities.
- Metrics: Top-1 sim ~0.16 on placeholder images → low.
- Findings: With random/placeholder images, scores are weak; model behaves as expected.
- Next: Add semantically relevant images (whales/animals); consider OCR fusion for text-heavy pics.

## Template for Implementation Phases
- Date: YYYY-MM-DD
- Phase: [Phase 1: OCR Pipeline | Phase 2: RRF Fusion | Phase 3: Testing]
- Context: [Current state, blockers, dependencies]
- Hypothesis: [Expected improvement, performance target]
- Method: [Implementation approach, tools used]
- Metrics: [Quantitative results - accuracy, latency, memory usage]
- Findings: [What worked, what didn't, surprises]
- Next: [Immediate next steps, risks to monitor]

## Metrics Tracking
- **Baseline (v0.1)**: Hit@3 ~20%, p50 latency ~0.8s, single-modal matching
- **Target (v0.2)**: Hit@3 ≥35%, p50 latency <1.2s, 4-channel hybrid retrieval
- **Progress**: Track each phase's contribution to overall improvement
