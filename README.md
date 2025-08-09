# CommonsAI: Community Q&A Index for Shared LLM Answers (MVP)

Not every question should be asked to a personal LLM. For public or academic questions, it’s often better to ask once, share openly, and let everyone reuse and build on the answer. CommonsAI is a community Q&A database: when you ask (text or image), we search for similar existing threads; if one fits, you jump into that thread, otherwise we open a new LLM‑assisted window and store it for future reuse.

### MVP scope (first step)
- Cross‑modal retrieval: encode both text and images into the same semantic space (OpenCLIP). A text question can find a matching image thread and vice versa.
- Thread selection flow: Ask → Retrieve Top‑K similar threads → you pick an existing thread or start a new LLM‑assisted thread.
- Local, inspectable storage (FAISS + .npy/.json/.jsonl) to iterate quickly.

### Why this matters
- Cost/energy: reuse existing high‑quality answers instead of regenerating them for each person.
- Collective intelligence: seeing others’ questions, LLM answers, and follow‑ups reveals depth of thinking and different angles that inspire new ideas.

## How it works (1 minute)
1) Put historical images in `data/images/`.
2) Build the index: encodes images → saves `img_embeds.npy` → builds `img.index` and `ids.json`.
3) Query: encode text → search FAISS → map indices to image IDs → return Top‑K and a HIT/MISS decision.
4) Update answers: community can improve answers via API; future queries reuse them.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Data
- Put images into `data/images/` (jpg/png)
- Build the index:
```bash
python -m app.build_index
```

## Run API
```bash
uvicorn app.server:app --reload --port 8000
```

### Endpoints
- GET `/health`
- POST `/query`  (form-data: `text`, optional `k`)
- POST `/update_answer`  (form-data: `img_id`, `answer`, `quality`)
- POST `/add_image`  (form-data: `file=@/abs/path.jpg`)
- POST `/set_tau`  (form-data: `tau`)

## Mapping
- `data/img_embeds.npy`: image vectors `(N, 512)`
- `data/ids.json`: row → image filename
- `data/img.index`: FAISS index
- `data/qa.jsonl`: answers/quality/tags

## CLI: Rank images
```bash
python scripts/rank.py --text "What is the largest mammal in the world?" --k 50
```

## Notes
- On macOS/ARM, compute text/image vectors first, then import FAISS in scripts to avoid ABI crashes.

## Research Timeline
See `RESEARCH_LOG.md`.
