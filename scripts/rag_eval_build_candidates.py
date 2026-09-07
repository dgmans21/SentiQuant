"""RAG quality eval, step 1: sample queries, retrieve top-5 via both embedding
search and a BM25 keyword baseline, and build a combined (query, candidate)
table for relevance judging.

BM25 uses character-bigram tokenization (no Korean morphological analyzer
installed) -- a common pragmatic choice for CJK text without konlpy/mecab,
and honestly it's exactly the kind of "dumb keyword matcher" we want as a
contrast baseline against the embedding model.
"""
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

N_QUERIES = 80
TOP_K = 5
SEED = 7

EMB_PATH = "data/processed/news_embeddings.npy"
META_PATH = "data/processed/news_embeddings_meta.csv"
OUT_PATH = "data/processed/rag_eval_candidates.csv"


def bigrams(text: str) -> list:
    text = (text or "").replace(" ", "")
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


if __name__ == "__main__":
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)
    meta["text"] = meta["title"].fillna("") + " " + meta["description"].fillna("")
    print(f"corpus size: {len(meta)}")

    rng = np.random.RandomState(SEED)
    query_idx = rng.choice(len(meta), size=N_QUERIES, replace=False)

    print("building BM25 index (character bigrams)...")
    tokenized_corpus = [bigrams(t) for t in meta["text"]]
    bm25 = BM25Okapi(tokenized_corpus)

    rows = []
    for qi in query_idx:
        q_row = meta.iloc[qi]
        q_text = q_row["text"]
        q_link = q_row["link"]

        # embedding method
        q_emb = emb[qi]
        sims = emb @ q_emb
        emb_order = np.argsort(-sims)
        emb_picked = [i for i in emb_order if i != qi][:TOP_K]

        # bm25 method
        bm25_scores = bm25.get_scores(bigrams(q_text))
        bm25_order = np.argsort(-bm25_scores)
        bm25_picked = [i for i in bm25_order if i != qi][:TOP_K]

        for rank, ci in enumerate(emb_picked):
            c = meta.iloc[ci]
            rows.append({
                "query_idx": qi, "query_stock": q_row["stock"], "query_title": q_row["title"],
                "query_desc": q_row["description"], "method": "embedding", "rank": rank + 1,
                "cand_idx": ci, "cand_title": c["title"], "cand_desc": c["description"],
                "cand_link": c["link"],
            })
        for rank, ci in enumerate(bm25_picked):
            c = meta.iloc[ci]
            rows.append({
                "query_idx": qi, "query_stock": q_row["stock"], "query_title": q_row["title"],
                "query_desc": q_row["description"], "method": "bm25", "rank": rank + 1,
                "cand_idx": ci, "cand_title": c["title"], "cand_desc": c["description"],
                "cand_link": c["link"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"queries: {N_QUERIES}, candidates per method: {TOP_K}, total rows: {len(out)}")
    print(f"saved to {OUT_PATH}")
