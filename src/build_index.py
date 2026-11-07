"""
Build gita_index_cache.pkl from the open vedicscriptures/bhagavad-gita dataset.

    git clone https://github.com/vedicscriptures/bhagavad-gita data/bhagavad-gita
    git -C data/bhagavad-gita checkout 43dfc8db815d01e15a347ea294b089334cf2aa17
    python src/build_index.py

Same steps as the live app: prepare_documents (drops chapter colophons),
all-MiniLM-L6-v2 embeddings, normalized, FAISS IndexFlatIP.
"""

import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.prepare_documents import prepare_documents  # noqa: E402

DATA_DIR = os.getenv("GITA_DATA", "data/bhagavad-gita/slok")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        sys.exit(f"No sloka files in {DATA_DIR} - clone the dataset first (see docstring).")
    raw = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            raw.append(json.load(f))

    docs = prepare_documents(raw)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [d["content"] for d in docs], normalize_embeddings=True, batch_size=32, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    with open("gita_index_cache.pkl", "wb") as f:
        pickle.dump({"index": faiss.serialize_index(index), "docs": docs}, f)
    print(f"{len(raw)} sloka files -> {len(docs)} verses indexed -> gita_index_cache.pkl")


if __name__ == "__main__":
    main()
