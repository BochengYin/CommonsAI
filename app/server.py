import faiss, numpy as np, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .embeddings import encode_text, encode_images
from .utils import *
from typing import Optional

app = FastAPI(title="LLM-Cache MVP", version="0.1")

# —— 启动时加载索引与元数据 ——
def load_state():
    ensure_dirs()
    if not INDEX_PATH.exists():
        return None, None, None, float(read_text(TAU_PATH, "0.30"))
    index = faiss.read_index(str(INDEX_PATH))
    ids = read_json(IDS_PATH, [])
    qa = read_jsonl(QA_PATH)
    tau = float(read_text(TAU_PATH, "0.30"))
    return index, ids, qa, tau

_index, _ids, _qa, _tau = load_state()

def refresh():
    global _index, _ids, _qa, _tau
    _index, _ids, _qa, _tau = load_state()

@app.get("/health")
def health():
    return {"ok": _index is not None, "tau": _tau, "num_images": len(_ids or [])}

@app.post("/set_tau")
def set_tau(tau: float = Form(...)):
    write_text(TAU_PATH, str(tau))
    refresh()
    return {"tau": _tau}

@app.post("/query")
def query(text: str = Form(...), k: int = Form(5)):
    if _index is None: 
        return JSONResponse({"error": "Index not built"}, status_code=400)
    qv = encode_text(text)  # (1, d)
    D, I = _index.search(qv.astype(np.float32), k)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    hits = []
    qa_map = {item["id"]: item for item in _qa}
    for sim, idx in zip(sims, idxs):
        img_id = _ids[idx]
        item = qa_map.get(img_id, {"answer": "", "quality": 0, "path": f"data/images/{img_id}"})
        hits.append({
            "img_id": img_id,
            "sim": float(sim),
            "answer": item.get("answer", ""),
            "quality": item.get("quality", 0),
            "path": item.get("path", f"data/images/{img_id}")
        })

    top = hits[0] if hits else None
    decision = None
    if top and top["sim"] >= _tau and top["answer"] and top["quality"] >= 3:
        decision = "HIT"   # 直接返回缓存答案
    else:
        decision = "MISS"  # 需要走 LLM（你可在前端/后端接 LLM，再写回）

    return {"decision": decision, "tau": _tau, "topk": hits}

@app.post("/update_answer")
def update_answer(img_id: str = Form(...), answer: str = Form(...), quality: int = Form(3)):
    # 更新/追加 qa.jsonl
    rows = read_jsonl(QA_PATH)
    found = False
    with QA_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("id") == img_id:
                r["answer"] = answer
                r["quality"] = quality
                found = True
            f.write((JSONResponse(r).body or b"").decode() + "\n")
    if not found:
        append_jsonl(QA_PATH, {
            "id": img_id, "type": "image",
            "path": f"data/images/{img_id}",
            "answer": answer, "quality": quality, "tags": []
        })
    refresh()
    return {"ok": True}

@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    ensure_dirs()
    ext = os.path.splitext(file.filename)[-1].lower()
    img_id = gen_id() + ext
    out_path = IMAGES_DIR / img_id
    with out_path.open("wb") as f:
        f.write(await file.read())

    # 编码并增量 add
    vec = encode_images([out_path])[0].astype(np.float32)
    if not INDEX_PATH.exists():
        # 首次：直接重建
        from .build_index import main as rebuild
        rebuild()
    else:
        index = faiss.read_index(str(INDEX_PATH))
        index.add(vec.reshape(1, -1))
        faiss.write_index(index, str(INDEX_PATH))
        ids = read_json(IDS_PATH, [])
        ids.append(img_id)
        write_json(IDS_PATH, ids)
        append_jsonl(QA_PATH, {
            "id": img_id, "type": "image",
            "path": f"data/images/{img_id}", "answer": "", "quality": 0, "tags": []
        })
    refresh()
    return {"id": img_id, "path": f"data/images/{img_id}"}


