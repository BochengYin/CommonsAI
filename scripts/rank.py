#!/usr/bin/env python3
import argparse, json
import numpy as np
from app.embeddings import encode_text
from app.utils import INDEX_PATH, IDS_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True, help='query text')
    ap.add_argument('--k', type=int, default=10, help='top-k to show')
    args = ap.parse_args()

    # Safe import order: encode first, then import faiss
    q = encode_text(args.text).astype(np.float32)

    import faiss  # lazy import to avoid ABI issues on some macOS/ARM setups
    index = faiss.read_index(str(INDEX_PATH))
    ids = json.load(open(IDS_PATH))

    k = min(args.k, len(ids))
    D, I = index.search(q, k)
    for rank, (sim, idx) in enumerate(zip(D[0], I[0]), 1):
        print(f"{rank:02d}  sim={sim:.4f}  id={ids[idx]}")


if __name__ == '__main__':
    main()
