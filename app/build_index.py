import os, json, numpy as np, glob
from tqdm import tqdm
from .embeddings import encode_images
from .utils import *

def main():
    ensure_dirs()
    # 收集 data/images 下所有图片
    img_paths = sorted(glob.glob(str(IMAGES_DIR / "*")))
    if not img_paths:
        print("⚠️ 请先把历史图片放到 data/images/ 再运行。")
        return

    # 仅保留常见可读格式，避免特殊格式触发底层库崩溃
    allowed_ext = {'.jpg', '.jpeg', '.png'}
    img_paths = [p for p in img_paths if os.path.splitext(p)[-1].lower() in allowed_ext]
    if not img_paths:
        print("⚠️ 未找到可支持的图片格式（仅支持 jpg/jpeg/png）。")
        return

    print(f"🔧 编码 {len(img_paths)} 张图片 ...")
    # 为了稳定性，逐张编码并收集，便于定位问题文件
    vec_list = []
    ok_ids = []
    for p in tqdm(img_paths, desc="encoding"):
        try:
            v = encode_images([p])  # (1, d)
            vec_list.append(v[0])
            ok_ids.append(os.path.basename(p))
        except Exception as e:
            print(f"跳过无法编码的图片: {p} -> {e}")
            continue
    if not vec_list:
        print("❌ 所有图片均编码失败。")
        return
    img_vecs = np.stack(vec_list, axis=0)
    np.save(EMB_PATH, img_vecs)

    # 避免与 PyTorch/OpenCLIP 的运行时冲突，延迟导入 faiss，并且在编码完成后再导入
    import faiss  # type: ignore
    d = img_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_vecs.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    ids = ok_ids
    write_json(IDS_PATH, ids)

    # 初始化/补齐 qa.jsonl（若无答案可先放空 answer）
    if not QA_PATH.exists():
        print("📝 初始化 qa.jsonl")
        for img_id in ids:
            append_jsonl(QA_PATH, {
                "id": img_id,
                "type": "image",
                "path": f"data/images/{img_id}",
                "answer": "",      # 你可以手动补充或在服务里更新
                "quality": 0,      # 社区质量分（0-5）
                "tags": []
            })

    # 默认阈值（可后续/服务端调整）
    if not TAU_PATH.exists():
        write_text(TAU_PATH, "0.30")

    print("✅ 索引构建完成：", INDEX_PATH)

if __name__ == "__main__":
    main()


