"""
Retriever for VedSanga's Gita search.

Pipeline:
    1. Synonym expansion (English <-> Sanskrit spiritual terms)
    2. Hybrid search: FAISS semantic (70%) + BM25 keyword (30%), top 20
    3. CrossEncoder rerank down to the final k

Embeddings: all-MiniLM-L6-v2. Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2.
"""

import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Several uvicorn workers share 2 vCPUs: one torch thread each is faster than
# every worker spawning a thread per core and fighting (measured 2.1 s vs 2.8 s).
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

# Lazy-loaded — avoids OOM on startup (Railway 1GB limit)
_embedding_model = None
_reranker = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ── FIX 2: Spiritual synonym expansion ───────────────────────────────────────
SYNONYMS = {
    "grief":      ["sorrow", "moha", "vishada", "lamentation", "despair"],
    "duty":       ["dharma", "karma", "svadharma", "righteous action"],
    "god":        ["krishna", "brahman", "atman", "supreme", "ishvara"],
    "soul":       ["atman", "self", "spirit", "jiva", "purusha"],
    "death":      ["mortal", "rebirth", "reincarnation", "indestructible"],
    "peace":      ["shanti", "equanimity", "moksha", "liberation"],
    "anger":      ["krodha", "passion", "wrath", "kama"],
    "mind":       ["manas", "buddhi", "intellect", "consciousness"],
    "action":     ["karma", "work", "deed", "nishkama karma"],
    "devotion":   ["bhakti", "surrender", "worship", "faith"],
    "knowledge":  ["jnana", "gyana", "wisdom", "truth"],
    "liberation": ["moksha", "mukti", "freedom", "self-realization"],
    "fear":       ["anxiety", "doubt", "worry", "abhaya"],
    "attachment": ["asakti", "clinging", "bondage", "maya"],
    "meditation": ["dhyana", "focus", "concentration", "samadhi"],
    "prayer":     ["worship", "bhakti", "devotion", "puja"],
    "war":        ["battle", "kurukshetra", "fight", "arjuna"],
    "yoga":       ["union", "path", "practice", "karma yoga"],
    "surrender":  ["sharanagati", "bhakti", "prapatti", "refuge"],
}

def expand_query(query: str) -> str:
    """Expand query with spiritual/Sanskrit synonyms before searching."""
    query_lower = query.lower()
    expansions = [query]
    for keyword, synonyms in SYNONYMS.items():
        if keyword in query_lower:
            expansions.extend(synonyms[:3])
    return " ".join(expansions)


# ── BM25 index (built once alongside FAISS) ───────────────────────────────────
_bm25_index = None
_bm25_docs  = None

def build_bm25(documents: List[Dict]):
    """Build BM25 keyword index from documents. Call once after vector store is built."""
    global _bm25_index, _bm25_docs
    tokenized = [doc["content"].lower().split() for doc in documents]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_docs  = documents
    print(f"✅ BM25 index built for {len(documents)} documents")


def _bm25_search(query: str, k: int = 20) -> Dict[int, float]:
    """BM25 keyword search — returns {doc_index: normalized_score}."""
    if _bm25_index is None:
        return {}
    scores = _bm25_index.get_scores(query.lower().split())
    max_s  = float(scores.max()) if scores.max() > 0 else 1.0
    norm   = scores / max_s
    top    = np.argsort(norm)[::-1][:k]
    return {int(i): float(norm[i]) for i in top}


def _faiss_search(query: str, index, k: int = 20) -> Dict[int, float]:
    """FAISS semantic search — returns {doc_index: score}."""
    qvec = _get_embedding_model().encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(qvec, k)
    return {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}


# ── FIX 3: Hybrid search ─────────────────────────────────────────────────────
def hybrid_search(
    query: str,
    index,
    documents: List[Dict],
    k: int = 20,
    w_sem: float = 0.7,
    w_bm25: float = 0.3
) -> List[Tuple[int, float]]:
    """Combine FAISS (semantic) + BM25 (keyword) scores."""
    sem  = _faiss_search(query, index, k)
    kw   = _bm25_search(query, k)
    all_idx = set(sem) | set(kw)
    combined = {
        i: sem.get(i, 0.0) * w_sem + kw.get(i, 0.0) * w_bm25
        for i in all_idx
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]


# ── FIX 4: CrossEncoder reranking ────────────────────────────────────────────
def rerank(
    query: str,
    candidates: List[Tuple[int, float]],
    documents: List[Dict],
    top_k: int
) -> List[Dict]:
    """Rerank candidates using CrossEncoder for final precision."""
    if not candidates:
        return []
    # Score against the verse's English meaning. "content" starts with the
    # Sanskrit shloka + transliteration, so its first 512 chars were mostly text
    # this English cross-encoder can't read — slow (7x) and less relevant.
    pairs  = [[query, (documents[i].get("metadata", {}).get("translation_en")
                       or documents[i]["content"])[:400]] for i, _ in candidates]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(
        zip([i for i, _ in candidates], scores),
        key=lambda x: x[1], reverse=True
    )
    return [documents[i] for i, _ in ranked[:top_k]]


# ── Main search function (drop-in replacement for old search()) ───────────────
def search(query: str, index, documents: List[Dict], k: int = 3) -> List[Dict]:
    """
    Full improved retrieval pipeline — drop-in replacement for old search().

    Steps:
        1. Expand query with spiritual synonyms
        2. Hybrid search: FAISS (70%) + BM25 (30%)
        3. CrossEncoder reranking for final top-k
    """
    # Step 1: Expand
    expanded = expand_query(query)

    # Step 2: Hybrid (get top 20 candidates)
    candidates = hybrid_search(expanded, index, documents, k=20)

    # Step 3: Rerank to final k
    results = rerank(query, candidates, documents, top_k=k)

    return results