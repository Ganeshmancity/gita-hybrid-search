# Gita hybrid search + eval

This is the search part of VedSanga, an app where people ask life questions and get answers from the Bhagavad Gita. The rest of the app is private, so this repo only has the retriever and the test I use to check it.

## How search works

1. The question gets a few synonyms added ("grief" also searches sorrow, moha, vishada).
2. BM25 (keyword) and FAISS (embeddings, `all-MiniLM-L6-v2`) each pull their top 20 verses. Scores are mixed 70% semantic, 30% keyword.
3. A CrossEncoder (`ms-marco-MiniLM-L-6-v2`) reranks those 20 and keeps the best 3.

Code: [`src/retriever.py`](src/retriever.py), documents built in [`src/prepare_documents.py`](src/prepare_documents.py).

## How I measured it

[`eval/retrieval_eval.py`](eval/retrieval_eval.py) has 48 plain-English questions. Each one is mapped to the verse a teacher would quote for it. I wrote them like people actually ask, mostly without the Sanskrit words, so the search can't just match keywords.

The script runs every question through four setups and checks where the right verse lands.

## Results

701 verses indexed, laptop CPU. Full output in [`eval/results.json`](eval/results.json).

| Setup | hit@1 | hit@3 | hit@10 | MRR | p50 | p95 |
|---|---|---|---|---|---|---|
| BM25 only | 0.58 | 0.69 | 0.81 | 0.65 | 9 ms | 13 ms |
| Embeddings only | 0.31 | 0.67 | 0.81 | 0.49 | 53 ms | 71 ms |
| Hybrid 70/30 | 0.71 | 0.77 | 0.88 | 0.76 | 64 ms | 73 ms |
| Hybrid + synonyms + rerank (live) | 0.75 | 0.85 | 0.90 | 0.80 | 1.8 s | 2.5 s |

What I took from it:

- Neither keyword nor embeddings is good enough alone. Mixing them is a clear jump.
- Embeddings alone were bad at putting the right verse first (31%). Exact words matter a lot in this text.
- The reranker adds about 8 points of hit@3 but almost all of the latency. Next step is to only rerank when the hybrid scores are close.
- Two earlier fixes came from measuring: reranking on the English translation instead of the Sanskrit-heavy text was about 7x faster, and one torch thread per worker on 2 vCPUs took latency from 2.8 s to 2.1 s.

7 questions still miss the top 3, mostly famous verses where my wording is far from the translation's wording. They're printed at the end of each run, so I can see exactly what's failing.

## Running it

Everything needed to get the same numbers is here. The verses come from the open [vedicscriptures/bhagavad-gita](https://github.com/vedicscriptures/bhagavad-gita) dataset (GPL-3), pinned to the commit I used. Library versions are pinned in `requirements.txt`.

```
git clone https://github.com/vedicscriptures/bhagavad-gita data/bhagavad-gita
git -C data/bhagavad-gita checkout 43dfc8db815d01e15a347ea294b089334cf2aa17
pip install -r requirements.txt   # Python 3.12
python src/build_index.py
python eval/retrieval_eval.py
```

The same steps run on GitHub Actions on every push ([retrieval-eval workflow](../../actions/workflows/eval.yml)), from a clean machine. Accuracy matches the table above exactly. Latency is lower there (rerank p50 ~0.45 s) because the runner CPU is faster than my laptop.
