import os, json, numpy as np, glob
from tqdm import tqdm
from .embeddings import encode_images
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

    print(f"🔧 Encoding {len(img_paths)} images ...")
    # Encode one-by-one for stability and better error localization
    vec_list = []
    ok_ids = []
    for p in tqdm(img_paths, desc="encoding"):
        try:
            v = encode_images([p])  # (1, d)
            vec_list.append(v[0])
            ok_ids.append(os.path.basename(p))
        except Exception as e:
            print(f"Skip problematic image: {p} -> {e}")
            continue
    if not vec_list:
        print("❌ All images failed to encode.")
        return
    img_vecs = np.stack(vec_list, axis=0)
    np.save(EMB_PATH, img_vecs)

    # Lazy-import faiss after encoding to avoid runtime conflicts with PyTorch/OpenCLIP
    import faiss  # type: ignore
    d = img_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_vecs.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    ids = ok_ids
    write_json(IDS_PATH, ids)

    # Initialize qa.jsonl if absent (use empty answer initially)
    if not QA_PATH.exists():
        print("📝 Initialize qa.jsonl")
        for img_id in ids:
            append_jsonl(QA_PATH, {
                "id": img_id,
                "type": "image",
                "path": f"data/images/{img_id}",
                "answer": "",      # you can update via service later
                "quality": 0,      # community quality score (0-5)
                "tags": []
            })

    # Default threshold (can be adjusted via service)
    if not TAU_PATH.exists():
        write_text(TAU_PATH, "0.30")

    print("✅ Index built:", INDEX_PATH)

if __name__ == "__main__":
    main()


