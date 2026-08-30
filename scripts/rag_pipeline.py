"""RAG demo: article text -> (1) klue-bert instant classification,
(2) retrieval of similar past articles + their actual multi-horizon excess returns.

Step 3 (Qwen-generated explanation) is intentionally NOT wired in here -- per the
"lean by default, Qwen only on an explicit 'more detail' action" design decision,
this fast path (steps 1-2) stands alone and is cheap enough to run on every request.
"""
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLASSIFIER_DIR = "models/klue-bert-qwen-sentiment-v4"
EMBEDDER_NAME = "jhgan/ko-sroberta-multitask"
EMB_PATH = "data/processed/news_embeddings.npy"
META_PATH = "data/processed/news_embeddings_meta.csv"
QWEN_MODEL_NAME = "unsloth/Qwen3-4B"

EXPLAIN_PROMPT = """다음은 한 뉴스 기사와 AI 분류 결과, 그리고 유사한 과거 기사들의 실제 결과다.

[기사]
{text}

[분류 결과] 논조: {label} (확신도 {confidence})

[유사 과거 사례 (제목 / 논조 / 발행 다음날 초과수익률 / 3거래일 후 초과수익률)]
{cases_text}

위 정보를 바탕으로 두 문단으로 답하라:
1. 이 기사가 왜 "{label}"로 분류됐는지 1~2문장으로 설명
2. 유사 과거 사례들의 수익률 패턴을 요약. 상관관계가 약하고 표본이 적을 수 있으니 확정적 예측처럼 말하지 말고, 참고용 통계로만 서술할 것"""


class SentimentRAG:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
        self.clf = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR).eval()
        self.embedder = SentenceTransformer(EMBEDDER_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        self.corpus_emb = np.load(EMB_PATH)
        self.meta = pd.read_csv(META_PATH)
        self.qwen_model = None
        self.qwen_tokenizer = None

    def _ensure_qwen_loaded(self):
        if self.qwen_model is not None:
            return
        from unsloth import FastModel
        self.qwen_model, self.qwen_tokenizer = FastModel.from_pretrained(
            model_name=QWEN_MODEL_NAME, max_seq_length=2048, load_in_4bit=True
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

    def classify(self, text: str):
        enc = self.tok(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.clf(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label = self.clf.config.id2label[int(probs.argmax())]
        return label, probs.max().item()

    def retrieve_similar(self, text: str, top_k: int = 5, exclude_link: str = None) -> pd.DataFrame:
        q_emb = self.embedder.encode([text], normalize_embeddings=True)[0]
        sims = self.corpus_emb @ q_emb
        order = np.argsort(-sims)
        rows = []
        for i in order:
            row = self.meta.iloc[i]
            if exclude_link is not None and row["link"] == exclude_link:
                continue
            rows.append({
                "similarity": round(float(sims[i]), 3),
                "stock": row["stock"], "title": row["title"],
                "qwen_label": row["qwen_label"],
                "excess_t1": row["excess_t1"], "excess_t3": row["excess_t3"],
                "excess_t5": row["excess_t5"], "excess_t10": row["excess_t10"],
            })
            if len(rows) >= top_k:
                break
        return pd.DataFrame(rows)

    def explain(self, text: str, label: str, confidence: float, similar_cases: pd.DataFrame) -> str:
        self._ensure_qwen_loaded()

        lines = []
        for _, r in similar_cases.iterrows():
            t1 = f"{r['excess_t1']:.2f}%" if pd.notna(r["excess_t1"]) else "N/A"
            t3 = f"{r['excess_t3']:.2f}%" if pd.notna(r["excess_t3"]) else "N/A"
            lines.append(f"- {r['title']} / {r['qwen_label']} / 다음날 {t1} / 3거래일 후 {t3}")
        cases_text = "\n".join(lines) if lines else "(유사 사례 없음)"

        prompt = EXPLAIN_PROMPT.format(text=text, label=label, confidence=confidence, cases_text=cases_text)
        chat_text = self.qwen_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=False,
        )
        inputs = self.qwen_tokenizer(chat_text, return_tensors="pt").to(self.qwen_model.device)
        with torch.no_grad():
            output = self.qwen_model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.qwen_tokenizer.pad_token_id,
            )
        gen_only = output[:, inputs["input_ids"].shape[1]:]
        response = self.qwen_tokenizer.decode(gen_only[0], skip_special_tokens=True)
        return response.strip()

    def analyze(self, text: str, top_k: int = 5) -> dict:
        label, conf = self.classify(text)
        similar = self.retrieve_similar(text, top_k=top_k)
        summary = {}
        for h in [1, 3, 5, 10]:
            col = f"excess_t{h}"
            valid = similar[col].dropna()
            summary[col] = round(valid.mean(), 3) if len(valid) > 0 else None
        return {"label": label, "confidence": round(conf, 3), "similar_cases": similar, "historical_avg": summary}


if __name__ == "__main__":
    import sys
    rag = SentimentRAG()
    text = sys.argv[1] if len(sys.argv) > 1 else "삼성전자, 3분기 영업이익 시장 예상치 상회하며 실적 개선세"

    result = rag.analyze(text)
    print(f"입력: {text}")
    print(f"논조: {result['label']} (확신도 {result['confidence']})")
    print()
    print("유사 과거 사례:")
    print(result["similar_cases"].to_string(index=False))
    print()
    print("유사 사례들의 평균 초과수익률 (참고용, 예측 아님):", result["historical_avg"])
