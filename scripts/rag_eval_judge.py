"""RAG quality eval, step 2: Qwen judges whether each (query, candidate) pair
is genuinely relevant (same company/event/topic) or just a coincidental
keyword/stock-name match -- same failure mode we saw with the "KT" -> kt wiz
baseball noise.
"""
import time
import pandas as pd
import torch
from unsloth import FastModel

IN_PATH = "data/processed/rag_eval_candidates.csv"
OUT_PATH = "data/processed/rag_eval_judged.csv"
BATCH_SIZE = 8

PROMPT_TEMPLATE = """다음은 검색 쿼리(원본 기사)와 검색 결과로 나온 후보 기사다. 후보가 쿼리와 실질적으로 관련된 내용(같은 기업/사건을 다루거나, 맥락상 의미 있게 연결됨)인지, 아니면 종목명 등이 우연히 겹칠 뿐 실제로는 무관한 내용(예: 같은 이름의 스포츠팀, 다른 주제)인지 판단하라.

쿼리 종목: {stock}
쿼리 제목: {q_title}
쿼리 요약: {q_desc}

후보 제목: {c_title}
후보 요약: {c_desc}

아래 형식으로만 답하라:
관련성: <관련|무관>"""

def build_prompt(row) -> str:
    return PROMPT_TEMPLATE.format(
        stock=row["query_stock"], q_title=row["query_title"], q_desc=row.get("query_desc", "") or "",
        c_title=row["cand_title"], c_desc=row.get("cand_desc", "") or "",
    )


def extract(text: str) -> str:
    # model often skips the "관련성:" prefix and just answers the bare word
    if "무관" in text:
        return "무관"
    if "관련" in text:
        return "관련"
    return "파싱실패"


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["query_desc"] = df["query_desc"].fillna("")
    df["cand_desc"] = df["cand_desc"].fillna("")
    print(f"판정 대상: {len(df)}건")

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-4B", max_seq_length=2048, load_in_4bit=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    results = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(r)}], add_generation_prompt=True,
                enable_thinking=False, tokenize=False,
            )
            for _, r in batch.iterrows()
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = output[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for r in responses:
            results.append((extract(r), r.strip().replace("\n", " ")))

        if (start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"{done}/{len(df)} | {rate:.2f}건/초")

    df["relevant"] = [r[0] for r in results]
    df["raw_response"] = [r[1] for r in results]
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n파싱실패: {(df['relevant'] == '파싱실패').sum()}건")
    print(f"saved to {OUT_PATH}")
