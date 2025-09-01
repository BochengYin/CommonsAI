import faiss, numpy as np, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .embeddings import encode_text, encode_images
from .hybrid_retrieval import get_hybrid_retriever, refresh_hybrid_retriever
from .bm25_search import get_bm25_engine, refresh_bm25_index
from .ocr import get_ocr_engine
from .utils import *
from .file_repository import FileRepository
from typing import Optional, List

app = FastAPI(title="CommonsAI", version="0.1")

# Initialize repository and load index at startup
_repo = FileRepository()
_hybrid_retriever = get_hybrid_retriever()
_bm25_engine = get_bm25_engine()
_ocr_engine = get_ocr_engine()

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
    refresh_hybrid_retriever()
    refresh_bm25_index()

@app.get("/health")
def health():
    hybrid_status = _hybrid_retriever.get_status()
    bm25_stats = _bm25_engine.get_stats()
    ocr_status = _ocr_engine.get_engine_status()
    
    return {
        "ok": _index is not None,
        "tau": _repo.get_tau(),
        "num_images": len(_repo.get_image_ids()),
        "hybrid_retrieval": hybrid_status,
        "bm25_search": bm25_stats,
        "ocr_engines": ocr_status,
        "version": "0.2.0-ocr"
    }

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

@app.post("/search_hybrid")
def search_hybrid(text: str = Form(...), k: int = Form(20), 
                 channels: Optional[str] = Form(None), debug: bool = Form(False)):
    """
    Hybrid search across multiple channels with RRF fusion.
    
    Args:
        text: Search query text
        k: Number of results to return per channel
        channels: Comma-separated list of channels (image_similarity,ocr_similarity,bm25_lexical)
        debug: Return detailed debug information
    """
    try:
        # Parse channels parameter
        channel_list = None
        if channels:
            channel_list = [c.strip() for c in channels.split(',')]
            # Validate channels
            valid_channels = {"image_similarity", "ocr_similarity", "bm25_lexical"}
            channel_list = [c for c in channel_list if c in valid_channels]
        
        if debug:
            # Return detailed debug information
            return _hybrid_retriever.search_with_debug(text, k)
        else:
            # Normal search with results formatting
            hybrid_results = _hybrid_retriever.search(text, k, channel_list)
            
            # Get QA data for results
            qa_records = _repo.get_all_qa()
            qa_map = {item["id"]: item for item in qa_records}
            
            # Format results
            formatted_results = []
            for result in hybrid_results:
                qa_data = qa_map.get(result.image_id, {})
                formatted_results.append({
                    "img_id": result.image_id,
                    "rrf_score": result.rrf_score,
                    "final_rank": result.final_rank,
                    "channels_matched": result.channels_matched,
                    "individual_scores": result.individual_scores,
                    "answer": qa_data.get("answer", ""),
                    "quality": qa_data.get("quality", 0),
                    "path": qa_data.get("path", f"data/images/{result.image_id}"),
                    "ocr_text": qa_data.get("ocr_text", ""),
                    "ocr_confidence": qa_data.get("ocr_confidence", 0.0)
                })
            
            # Determine HIT/MISS decision based on top result
            top_result = formatted_results[0] if formatted_results else None
            tau = _repo.get_tau()
            
            if (top_result and top_result["rrf_score"] >= tau and 
                top_result["answer"] and top_result["quality"] >= 3):
                decision = "HIT"
            else:
                decision = "MISS"
            
            return {
                "decision": decision,
                "tau": tau,
                "search_type": "hybrid",
                "channels_used": channel_list or ["image_similarity", "ocr_similarity", "bm25_lexical"],
                "topk": formatted_results
            }
            
    except Exception as e:
        return JSONResponse({"error": f"Hybrid search failed: {str(e)}"}, status_code=500)

@app.post("/search_bm25")
def search_bm25_only(text: str = Form(...), k: int = Form(10)):
    """BM25-only lexical search for OCR text."""
    try:
        results = _bm25_engine.search_with_metadata(text, k)
        
        # Get QA data for results
        qa_records = _repo.get_all_qa()
        qa_map = {item["id"]: item for item in qa_records}
        
        formatted_results = []
        for result in results:
            qa_data = qa_map.get(result["image_id"], {})
            formatted_results.append({
                "img_id": result["image_id"],
                "bm25_score": result["bm25_score"],
                "matched_tokens": result["matched_tokens"],
                "match_count": result["match_count"],
                "answer": qa_data.get("answer", ""),
                "quality": qa_data.get("quality", 0),
                "path": qa_data.get("path", f"data/images/{result['image_id']}"),
                "ocr_text": qa_data.get("ocr_text", "")
            })
        
        return {
            "search_type": "bm25_lexical",
            "results": formatted_results,
            "total_matches": len(formatted_results)
        }
        
    except Exception as e:
        return JSONResponse({"error": f"BM25 search failed: {str(e)}"}, status_code=500)

@app.post("/search_ocr_similarity")  
def search_ocr_similarity_only(text: str = Form(...), k: int = Form(10)):
    """OCR text similarity search using OpenCLIP embeddings."""
    try:
        # Use hybrid retriever but only OCR similarity channel
        hybrid_results = _hybrid_retriever.search(text, k, channels=["ocr_similarity"])
        
        # Get QA data for results
        qa_records = _repo.get_all_qa()
        qa_map = {item["id"]: item for item in qa_records}
        
        formatted_results = []
        for result in hybrid_results:
            qa_data = qa_map.get(result.image_id, {})
            ocr_score = result.individual_scores.get("ocr_similarity", 0.0)
            
            formatted_results.append({
                "img_id": result.image_id,
                "ocr_similarity_score": ocr_score,
                "answer": qa_data.get("answer", ""),
                "quality": qa_data.get("quality", 0), 
                "path": qa_data.get("path", f"data/images/{result.image_id}"),
                "ocr_text": qa_data.get("ocr_text", ""),
                "ocr_confidence": qa_data.get("ocr_confidence", 0.0)
            })
        
        return {
            "search_type": "ocr_similarity",
            "results": formatted_results,
            "total_matches": len(formatted_results)
        }
        
    except Exception as e:
        return JSONResponse({"error": f"OCR similarity search failed: {str(e)}"}, status_code=500)

@app.get("/ocr_status")
def ocr_status():
    """Get OCR engine status and statistics."""
    try:
        ocr_status = _ocr_engine.get_engine_status()
        
        # Get OCR statistics from QA data
        qa_records = _repo.get_all_qa()
        ocr_stats = {
            "total_images": len(qa_records),
            "images_with_ocr": len([r for r in qa_records if r.get("ocr_text", "").strip()]),
            "avg_ocr_confidence": 0.0,
            "languages": set(),
            "engines_used": set()
        }
        
        confidences = []
        for record in qa_records:
            if record.get("ocr_confidence"):
                confidences.append(record["ocr_confidence"])
            if record.get("ocr_language"):
                ocr_stats["languages"].add(record["ocr_language"])
            if record.get("ocr_engine"):
                ocr_stats["engines_used"].add(record["ocr_engine"])
        
        if confidences:
            ocr_stats["avg_ocr_confidence"] = sum(confidences) / len(confidences)
        
        ocr_stats["languages"] = list(ocr_stats["languages"])
        ocr_stats["engines_used"] = list(ocr_stats["engines_used"])
        
        return {
            "engine_availability": ocr_status,
            "corpus_stats": ocr_stats
        }
        
    except Exception as e:
        return JSONResponse({"error": f"OCR status failed: {str(e)}"}, status_code=500)


