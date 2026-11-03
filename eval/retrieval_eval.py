"""
Retrieval evaluation for the VedSanga Gita search.

Each question is a plain-language paraphrase (no verse numbers, mostly no
Sanskrit keywords) mapped to the verse(s) a scholar would cite for it.
Measures hit@1 / hit@3 / hit@10, MRR and latency for four configurations:
  1. BM25 only           2. Semantic (FAISS) only
  3. Hybrid (70/30)      4. Hybrid + synonym expansion + CrossEncoder rerank (production)

Run from the repo root:  python eval/retrieval_eval.py
"""

import json
import os
import pickle
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss  # noqa: E402
from src import retriever as R  # noqa: E402

# (question, [acceptable "chapter.verse" ids])
GOLD = [
    ("Do your work but don't be attached to the results", ["2.47"]),
    ("The soul is never born and never dies", ["2.20"]),
    ("Weapons cannot cut the soul and fire cannot burn it", ["2.23", "2.24"]),
    ("Just as a person throws away old clothes and puts on new ones, the soul takes a new body", ["2.22"]),
    ("Whenever righteousness declines I take birth on earth", ["4.7"]),
    ("I come age after age to protect the good and destroy evil", ["4.8"]),
    ("Give up all other duties and just surrender to me, I will free you from sin", ["18.66"]),
    ("If someone offers me a leaf, a flower, fruit or water with love, I accept it", ["9.26"]),
    ("It is better to do your own duty badly than someone else's duty well", ["3.35", "18.47"]),
    ("Anger leads to confusion, confusion to loss of memory, and then the person is ruined", ["2.63"]),
    ("Thinking about sense objects creates attachment, attachment creates desire, desire creates anger", ["2.62"]),
    ("Raise yourself by your own mind, the mind can be your friend or your enemy", ["6.5", "6.6"]),
    ("The mind is restless and hard to control but it can be tamed by practice and detachment", ["6.35", "6.34"]),
    ("Yoga is skill in action", ["2.50"]),
    ("Staying balanced in success and failure is called yoga", ["2.48"]),
    ("I am time, the mighty destroyer of the worlds", ["11.32"]),
    ("Arjuna's limbs give way and his mouth dries up when he sees his relatives on the battlefield", ["1.28", "1.29"]),
    ("Arjuna drops his bow and sits down in the chariot overcome with sorrow", ["1.47"]),
    ("Death is certain for anyone who is born, so you should not grieve over the inevitable", ["2.27"]),
    ("Desire and anger born of passion are the great enemy of man", ["3.37"]),
    ("There are three gates to hell: lust, anger and greed", ["16.21"]),
    ("Whatever a great person does, ordinary people follow", ["3.21"]),
    ("Always think of me, be my devotee, worship me and bow to me, and you will come to me", ["18.65", "9.34"]),
    ("Of all yogis, the one who worships me with faith and love is the highest", ["6.47"]),
    ("Heat and cold, pleasure and pain come and go; endure them patiently", ["2.14"]),
    ("Treat pleasure and pain, gain and loss, victory and defeat as the same and then fight", ["2.38"]),
    ("Foods that increase life, strength, health and happiness are dear to the good", ["17.8"]),
    ("Like a lamp in a windless place that does not flicker, the yogi's mind is steady", ["6.19"]),
    ("One who sees inaction in action and action in inaction is truly wise", ["4.18"]),
    ("The fire of knowledge burns all karma to ashes", ["4.37"]),
    ("Nothing in this world purifies like knowledge", ["4.38"]),
    ("Among the Vedas I am the Sama Veda and among the senses I am the mind", ["10.22"]),
    ("Whatever a person thinks of at the moment of death, that is what they attain", ["8.6"]),
    ("Whoever remembers me while leaving the body at death comes to me", ["8.5"]),
    ("One who hates no being, is friendly and compassionate, is dear to me", ["12.13"]),
    ("The eternal tree with roots above and branches below whose leaves are the Vedas", ["15.1"]),
    ("I am seated in the hearts of all beings and from me come memory and knowledge", ["15.15"]),
    ("That supreme abode is not lit by the sun, the moon or fire", ["15.6"]),
    ("Everyone's faith matches their own nature", ["17.3"]),
    ("Charity given to a worthy person at the right time and place without expecting return is good charity", ["17.20"]),
    ("Wherever Krishna and Arjuna are, there will be prosperity, victory and righteousness", ["18.78"]),
    ("My delusion is gone, my doubts are cleared and I will do as you say", ["18.73"]),
    ("For those who worship me with single-minded devotion, I bring what they lack and protect what they have", ["9.22"]),
    ("Yoga removes sorrow for one who is moderate in eating, sleeping, work and recreation", ["6.17"]),
    ("A sage of steady wisdom is not shaken by sorrow, does not crave pleasure, and is free from fear and anger", ["2.56"]),
    ("Like a tortoise pulling in its limbs, the wise withdraw their senses from objects", ["2.58"]),
    ("Like rivers entering the ocean which stays calm, desires enter the wise without disturbing them", ["2.70"]),
    ("Even a little effort on this path saves you from great fear", ["2.40"]),
]


def vid(doc):
    m = doc.get("metadata", {})
    return f"{m.get('chapter')}.{m.get('verse')}"


def rank_of(ids, gold):
    for i, v in enumerate(ids, 1):
        if v in gold:
            return i
    return None


def run_config(name, fn, docs):
    ranks, times = [], []
    misses = []
    for q, gold in GOLD:
        t0 = time.perf_counter()
        idxs = fn(q)
        times.append(time.perf_counter() - t0)
        ids = [vid(docs[i]) for i in idxs]
        r = rank_of(ids, gold)
        ranks.append(r)
        if r is None or r > 3:
            misses.append((q, gold, ids[:3]))
    n = len(GOLD)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    times_ms = sorted(t * 1000 for t in times)
    p95 = times_ms[int(0.95 * (n - 1))]
    return {
        "config": name,
        "hit@1": round(hit(1), 3),
        "hit@3": round(hit(3), 3),
        "hit@10": round(hit(10), 3),
        "mrr": round(mrr, 3),
        "p50_ms": round(statistics.median(times_ms)),
        "p95_ms": round(p95),
        "misses_top3": misses,
    }


def main():
    with open("gita_index_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    index = faiss.deserialize_index(cache["index"])
    docs = cache["docs"]
    R.build_bm25(docs)

    known = {vid(d) for d in docs}
    missing = [g for _, gs in GOLD for g in gs if g not in known]
    if missing:
        print("WARNING: gold ids not in index:", missing)

    # warm up model loads so latency reflects steady-state queries
    R.search("warm up", index, docs, k=3)

    configs = [
        ("bm25_only", lambda q: [i for i, _ in sorted(R._bm25_search(q, 20).items(), key=lambda x: -x[1])][:10]),
        ("semantic_only", lambda q: [i for i, _ in sorted(R._faiss_search(q, index, 20).items(), key=lambda x: -x[1])][:10]),
        ("hybrid_70_30", lambda q: [i for i, _ in R.hybrid_search(q, index, docs, k=20)][:10]),
        ("production_hybrid_expand_rerank", lambda q: _prod(q, index, docs)),
    ]
    results = [run_config(n, fn, docs) for n, fn in configs]

    print(f"\nVedSanga retrieval eval — {len(GOLD)} questions, {len(docs)} verses indexed\n")
    print(f"{'config':34} {'hit@1':>6} {'hit@3':>6} {'hit@10':>7} {'MRR':>6} {'p50ms':>6} {'p95ms':>6}")
    for r in results:
        print(f"{r['config']:34} {r['hit@1']:>6} {r['hit@3']:>6} {r['hit@10']:>7} {r['mrr']:>6} {r['p50_ms']:>6} {r['p95_ms']:>6}")

    prod = results[-1]
    print(f"\nProduction misses (gold not in top 3): {len(prod['misses_top3'])}")
    for q, gold, top in prod["misses_top3"]:
        print(f"  - {q}\n      want {gold}, got {top}")

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump({"n_questions": len(GOLD), "n_docs": len(docs), "results": results}, f, indent=2, ensure_ascii=False)
    print("\nSaved eval/results.json")


def _prod(q, index, docs):
    """Production pipeline, but returning ranked indices (top 10) for scoring."""
    expanded = R.expand_query(q)
    cands = R.hybrid_search(expanded, index, docs, k=20)
    pairs = [[q, (docs[i].get("metadata", {}).get("translation_en") or docs[i]["content"])[:400]] for i, _ in cands]
    scores = R._get_reranker().predict(pairs)
    ranked = sorted(zip([i for i, _ in cands], scores), key=lambda x: -x[1])
    return [i for i, _ in ranked][:10]


if __name__ == "__main__":
    main()
