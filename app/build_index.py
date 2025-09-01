import os, json, numpy as np, glob
from tqdm import tqdm
from .embeddings import encode_images, encode_ocr_text, create_embedding_metadata
from .ocr import get_ocr_engine
from .utils import *

def main():
    ensure_dirs()
    # Collect all images under data/images
    img_paths = sorted(glob.glob(str(IMAGES_DIR / "*")))
    if not img_paths:
        print("⚠️ Please put historical images into data/images/ and re-run.")
        return

    # Keep common formats only to avoid decoder/runtime issues
    allowed_ext = {'.jpg', '.jpeg', '.png'}
    img_paths = [p for p in img_paths if os.path.splitext(p)[-1].lower() in allowed_ext]
    if not img_paths:
        print("⚠️ No supported image format found (jpg/jpeg/png only).")
        return

    print(f"🔧 Processing {len(img_paths)} images ...")
    
    # Initialize OCR engine
    ocr_engine = get_ocr_engine(prefer_gpu=True, confidence_threshold=0.5)
    print(f"🔍 OCR engines available: {ocr_engine.get_engine_status()}")
    
    # Phase 1: Extract OCR text from all images
    print("📝 Phase 1: Extracting OCR text...")
    ocr_results = []
    ocr_texts = {}
    
    for p in tqdm(img_paths, desc="OCR extraction"):
        try:
            img_id = os.path.basename(p)
            ocr_result = ocr_engine.extract_text(p)
            ocr_results.append(ocr_result)
            ocr_texts[img_id] = ocr_result.to_dict()
        except Exception as e:
            print(f"OCR failed for {p}: {e}")
            # Create empty OCR result for consistency
            from .ocr import OCRResult
            empty_result = OCRResult("", 0.0, "unknown", "error")
            ocr_results.append(empty_result)
            ocr_texts[os.path.basename(p)] = empty_result.to_dict()
    
    # Save OCR results
    ocr_path = DATA / "ocr_text.json"
    write_json(ocr_path, ocr_texts)
    print(f"💾 OCR results saved to {ocr_path}")
    
    # Phase 2: Generate image embeddings
    print("🖼️ Phase 2: Generating image embeddings...")
    img_vec_list = []
    ok_ids = []
    ok_ocr_results = []
    
    for i, p in enumerate(tqdm(img_paths, desc="Image encoding")):
        try:
            v = encode_images([p])  # (1, d)
            img_vec_list.append(v[0])
            ok_ids.append(os.path.basename(p))
            ok_ocr_results.append(ocr_results[i])
        except Exception as e:
            print(f"Skip problematic image: {p} -> {e}")
            continue
    
    if not img_vec_list:
        print("❌ All images failed to encode.")
        return
    
    img_vecs = np.stack(img_vec_list, axis=0)
    np.save(EMB_PATH, img_vecs)
    
    # Phase 3: Generate OCR text embeddings
    print("📝 Phase 3: Generating OCR text embeddings...")
    ocr_vecs = encode_ocr_text(ok_ocr_results)
    
    ocr_emb_path = DATA / "ocr_embeds.npy"
    np.save(ocr_emb_path, ocr_vecs)
    print(f"💾 OCR embeddings saved to {ocr_emb_path}")
    
    # Phase 4: Build FAISS index for images (keep existing structure)
    print("🔍 Phase 4: Building FAISS image index...")
    # Lazy-import faiss after encoding to avoid runtime conflicts with PyTorch/OpenCLIP
    import faiss  # type: ignore
    d = img_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_vecs.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    # Phase 5: Build BM25 index for OCR text
    print("🔤 Phase 5: Building BM25 text index...")
    build_bm25_index(ok_ocr_results, ok_ids)
    
    # Save image IDs
    write_json(IDS_PATH, ok_ids)

    # Phase 6: Initialize/update qa.jsonl with OCR information
    print("📋 Phase 6: Updating qa.jsonl with OCR data...")
    if not QA_PATH.exists():
        print("📝 Initialize qa.jsonl")
        for i, img_id in enumerate(ok_ids):
            ocr_result = ok_ocr_results[i]
            append_jsonl(QA_PATH, {
                "id": img_id,
                "type": "image",
                "path": f"data/images/{img_id}",
                "answer": "",      # you can update via service later
                "quality": 0,      # community quality score (0-5)
                "tags": [],
                "ocr_text": ocr_result.text,
                "ocr_confidence": ocr_result.confidence,
                "ocr_language": ocr_result.language,
                "ocr_engine": ocr_result.engine,
                "text_hash": ocr_result.hash
            })
    else:
        # Update existing records with OCR data
        print("🔄 Updating existing qa.jsonl with OCR data")
        existing_records = list(read_jsonl(QA_PATH))
        existing_ids = {r["id"] for r in existing_records}
        
        # Update existing records and add new ones
        updated_records = []
        for record in existing_records:
            if record["id"] in ocr_texts:
                ocr_data = ocr_texts[record["id"]]
                record.update({
                    "ocr_text": ocr_data["text"],
                    "ocr_confidence": ocr_data["confidence"],
                    "ocr_language": ocr_data["language"],
                    "ocr_engine": ocr_data["engine"],
                    "text_hash": ocr_data["hash"]
                })
            updated_records.append(record)
        
        # Add new images not in existing records
        for i, img_id in enumerate(ok_ids):
            if img_id not in existing_ids:
                ocr_result = ok_ocr_results[i]
                updated_records.append({
                    "id": img_id,
                    "type": "image", 
                    "path": f"data/images/{img_id}",
                    "answer": "",
                    "quality": 0,
                    "tags": [],
                    "ocr_text": ocr_result.text,
                    "ocr_confidence": ocr_result.confidence,
                    "ocr_language": ocr_result.language,
                    "ocr_engine": ocr_result.engine,
                    "text_hash": ocr_result.hash
                })
        
        # Rewrite qa.jsonl
        QA_PATH.unlink()  # Remove old file
        for record in updated_records:
            append_jsonl(QA_PATH, record)

    # Default threshold (can be adjusted via service)
    if not TAU_PATH.exists():
        write_text(TAU_PATH, "0.30")
    
    # Print summary statistics
    ocr_stats = analyze_ocr_results(ok_ocr_results)
    print(f"\n📊 OCR Statistics:")
    print(f"   Images with text: {ocr_stats['images_with_text']}/{len(ok_ocr_results)}")
    print(f"   Average confidence: {ocr_stats['avg_confidence']:.2f}")
    print(f"   Engines used: {ocr_stats['engines_used']}")
    print(f"   Languages detected: {ocr_stats['languages']}")
    
    print("✅ Multi-modal index built successfully!")
    print(f"   📁 Image embeddings: {EMB_PATH}")
    print(f"   📁 OCR embeddings: {ocr_emb_path}")
    print(f"   📁 FAISS index: {INDEX_PATH}")
    print(f"   📁 BM25 index: {DATA / 'bm25.pkl'}")

def build_bm25_index(ocr_results, image_ids):
    """Build BM25 index for OCR text search."""
    try:
        from rank_bm25 import BM25Okapi
        import pickle
        
        # Prepare documents for BM25 (tokenize)
        documents = []
        valid_indices = []
        
        for i, ocr_result in enumerate(ocr_results):
            if ocr_result.confidence >= 0.5 and len(ocr_result.text.strip()) > 0:
                # Simple tokenization (can be improved with proper NLP)
                tokens = ocr_result.text.lower().split()
                if tokens:  # Only add non-empty documents
                    documents.append(tokens)
                    valid_indices.append(i)
        
        if documents:
            bm25 = BM25Okapi(documents)
            
            # Save BM25 index and metadata
            bm25_data = {
                "bm25": bm25,
                "image_indices": valid_indices,  # Maps BM25 doc index to image index
                "image_ids": [image_ids[i] for i in valid_indices]  # Corresponding image IDs
            }
            
            bm25_path = DATA / "bm25.pkl"
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            print(f"💾 BM25 index built with {len(documents)} text documents")
        else:
            print("⚠️ No valid OCR text found for BM25 index")
            
    except ImportError:
        print("⚠️ rank-bm25 not available, skipping BM25 index")

def analyze_ocr_results(ocr_results):
    """Analyze OCR results and return statistics."""
    stats = {
        "images_with_text": 0,
        "avg_confidence": 0.0,
        "engines_used": set(),
        "languages": set(),
        "total_chars": 0
    }
    
    confidences = []
    for result in ocr_results:
        if len(result.text.strip()) > 0:
            stats["images_with_text"] += 1
            stats["total_chars"] += len(result.text)
        
        confidences.append(result.confidence)
        stats["engines_used"].add(result.engine)
        stats["languages"].add(result.language)
    
    stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    stats["engines_used"] = list(stats["engines_used"])
    stats["languages"] = list(stats["languages"])
    
    return stats

if __name__ == "__main__":
    main()


