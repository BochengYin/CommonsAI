# Shared LLM Answer Cache (MVP)

A local prototype to validate a shared, searchable, cacheable LLM Q&A system: users ask (text or image), the system first searches previous similar items; on hit, it returns the curated cached answer; on miss, it drafts with an LLM and lets the community improve and store the answer.

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
