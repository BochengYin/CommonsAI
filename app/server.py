import faiss, numpy as np, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .embeddings import encode_text, encode_images
from .utils import *
from .file_repository import FileRepository
from typing import Optional

app = FastAPI(title="CommonsAI", version="0.1")

# Initialize repository and load index at startup
_repo = FileRepository()

def load_index():
    ensure_dirs()
    if not INDEX_PATH.exists():
        return None
    return faiss.read_index(str(INDEX_PATH))

_index = load_index()

def refresh():
    global _index
    _index = load_index()
    _repo.refresh()

@app.get("/health")
def health():
    return {"ok": _index is not None, "tau": _repo.get_tau(), "num_images": len(_repo.get_image_ids())}

@app.post("/set_tau")
def set_tau(tau: float = Form(...)):
    _repo.set_tau(tau)
    return {"tau": _repo.get_tau()}

@app.post("/query")
def query(text: str = Form(...), k: int = Form(5)):
    if _index is None: 
        return JSONResponse({"error": "Index not built"}, status_code=400)
    qv = encode_text(text)  # (1, d)
    D, I = _index.search(qv.astype(np.float32), k)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    hits = []
    ids = _repo.get_image_ids()
    qa_records = _repo.get_all_qa()
    qa_map = {item["id"]: item for item in qa_records}
    
    for sim, idx in zip(sims, idxs):
        img_id = ids[idx]
        item = qa_map.get(img_id, {"answer": "", "quality": 0, "path": f"data/images/{img_id}"})
        hits.append({
            "img_id": img_id,
            "sim": float(sim),
            "answer": item.get("answer", ""),
            "quality": item.get("quality", 0),
            "path": item.get("path", f"data/images/{img_id}")
        })

    top = hits[0] if hits else None
    tau = _repo.get_tau()
    decision = None
    if top and top["sim"] >= tau and top["answer"] and top["quality"] >= 3:
        decision = "HIT"   # return cached answer directly
    else:
        decision = "MISS"  # need to call LLM, then write back via update

    return {"decision": decision, "tau": tau, "topk": hits}

@app.post("/update_answer")
def update_answer(img_id: str = Form(...), answer: str = Form(...), quality: int = Form(3)):
    _repo.upsert_qa(img_id, answer, quality)
    return {"ok": True}

@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    ensure_dirs()
    ext = os.path.splitext(file.filename)[-1].lower()
    img_id = gen_id() + ext
    out_path = IMAGES_DIR / img_id
    with out_path.open("wb") as f:
        f.write(await file.read())

    # Encode and incrementally add
    vec = encode_images([out_path])[0].astype(np.float32)
    if not INDEX_PATH.exists():
        # First time: rebuild
        from .build_index import main as rebuild
        rebuild()
    else:
        index = faiss.read_index(str(INDEX_PATH))
        index.add(vec.reshape(1, -1))
        faiss.write_index(index, str(INDEX_PATH))
        _repo.add_image_id(img_id)
        _repo.upsert_qa(img_id, "", 0, f"data/images/{img_id}")
    refresh()
    return {"id": img_id, "path": f"data/images/{img_id}"}


