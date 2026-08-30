"""RAG step 1: embed all collected news (title+description) for similarity search.

Uses a lightweight Korean sentence-embedding model (jhgan/ko-sroberta-multitask,
~440MB, 768-dim) -- separate from klue-bert-qwen-sentiment (classification) and
Qwen3-4B (generation). Embeddings + metadata (stock, link, qwen_label, multi-horizon
excess returns where available) are saved together so the retrieval step can look
up "what actually happened" for any retrieved similar article.
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sroberta-multitask"
OUT_EMB_PATH = "data/processed/news_embeddings.npy"
OUT_META_PATH = "data/processed/news_embeddings_meta.csv"


def load_corpus() -> pd.DataFrame:
    qwen = pd.read_csv("data/processed/news_qwen_labeled.csv")
    qwen = qwen[qwen["qwen_label"].isin(["긍정", "부정", "중립"])].copy()

    horizon_path = "data/processed/news_multi_horizon.csv"
    try:
        horizon = pd.read_csv(horizon_path)
        keep = [c for c in horizon.columns if c.startswith("excess_t")]
        qwen = qwen.merge(horizon[["link"] + keep], on="link", how="left")
    except FileNotFoundError:
        pass

    qwen["text"] = qwen["title"].fillna("") + " " + qwen["description"].fillna("")
    return qwen


if __name__ == "__main__":
    df = load_corpus()
    print(f"임베딩 대상: {len(df)}건")

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    embeddings = model.encode(
        df["text"].tolist(), batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    np.save(OUT_EMB_PATH, embeddings.astype("float32"))
    meta_cols = [c for c in df.columns if c != "text"]
    df[meta_cols].to_csv(OUT_META_PATH, index=False, encoding="utf-8-sig")

    print(f"임베딩 shape: {embeddings.shape}")
    print(f"saved to {OUT_EMB_PATH}, {OUT_META_PATH}")
