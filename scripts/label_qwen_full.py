"""Full-scale Qwen3-4B sentiment labeling with batched generation + checkpointing.

Processes data/processed/news_labeled_excess_v2.csv (11,188 rows) in batches
for speed, writing results incrementally every CHUNK_SIZE rows so progress
survives an interruption -- on restart it skips rows already labeled (by link).
"""
import os
import re
import sys
import time
import pandas as pd
import torch
from unsloth import FastModel

DATA_PATH = "data/processed/news_labeled_excess_v5.csv"
OUT_PATH = "data/processed/news_qwen_labeled.csv"
BATCH_SIZE = 8
CHUNK_SIZE = 1000
TIME_LIMIT_SECONDS = 600  # stop after ~10 minutes; resumable via checkpoint
MAX_ITEMS_THIS_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else None  # overrides time limit if set

PROMPT_TEMPLATE = """다음은 한국 주식 종목 관련 뉴스 제목과 요약이다. 아래 순서로 판단하라.

1단계: 이 기사가 실제로 "{stock}"이라는 기업/종목 자체의 사업, 실적, 주가, 경영 활동과 관련이 있는가?
   - 종목명이 우연히 언급될 뿐 실제 내용은 무관한 주제(예: 같은 이름의 스포츠팀, 다른 기업 소식에 곁가지로 등장, 오탐)라면 "무관"으로 답하고 종료하라.
2단계: 관련이 있다면, 이 기사가 "{stock}"에 대해 전달하는 논조가 긍정적인지 부정적인지 중립적인지 판단하라.
   - 객관적 사실 전달이라도 내용이 실적 부진, 하락, 리스크 등을 담고 있으면 부정으로, 성장, 호실적, 상승 등을 담고 있으면 긍정으로 판단하라.

종목: {stock}
제목: {title}
요약: {description}

아래 형식으로만 답하라:
라벨: <긍정|부정|중립|무관>
이유: <한 줄 이유>"""

LABEL_RE = re.compile(r"라벨:\s*(긍정|부정|중립|무관)")


def build_prompt(row) -> str:
    return PROMPT_TEMPLATE.format(
        stock=row["stock"], title=row["title"], description=row.get("description", "") or ""
    )


def extract_label(text: str) -> str:
    m = LABEL_RE.search(text)
    return m.group(1) if m else "파싱실패"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["description"] = df["description"].fillna("")

    done_links = set()
    if os.path.exists(OUT_PATH):
        done_links = set(pd.read_csv(OUT_PATH)["link"])
        print(f"resuming: {len(done_links)} rows already labeled")

    todo = df[~df["link"].isin(done_links)].reset_index(drop=True)
    print(f"remaining: {len(todo)} / {len(df)}")

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-4B", max_seq_length=2048, load_in_4bit=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_start = time.time()
    n_done_this_run = 0
    buffer = []

    time_limit_hit = False
    for batch_start in range(0, len(todo), BATCH_SIZE):
        if MAX_ITEMS_THIS_RUN is not None:
            if n_done_this_run >= MAX_ITEMS_THIS_RUN:
                break
        elif time.time() - t_start > TIME_LIMIT_SECONDS:
            time_limit_hit = True
            break
        batch = todo.iloc[batch_start:batch_start + BATCH_SIZE]
        prompts = [build_prompt(r) for _, r in batch.iterrows()]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                enable_thinking=False, tokenize=False,
            )
            for p in prompts
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_only = output[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for (_, row), resp in zip(batch.iterrows(), responses):
            buffer.append({
                "stock": row["stock"],
                "sector": row["sector"],
                "news_date": row["news_date"],
                "title": row["title"],
                "description": row["description"],
                "link": row["link"],
                "price_based_label": row.get("label", None),
                "qwen_label": extract_label(resp),
                "qwen_response": resp.strip().replace("\n", " "),
            })

        n_done_this_run += len(batch)

        if len(buffer) >= CHUNK_SIZE or batch_start + BATCH_SIZE >= len(todo):
            chunk_df = pd.DataFrame(buffer)
            write_header = not os.path.exists(OUT_PATH)
            chunk_df.to_csv(OUT_PATH, mode="a", index=False, header=write_header, encoding="utf-8-sig")
            elapsed = time.time() - t_start
            rate = n_done_this_run / elapsed
            remaining = len(todo) - n_done_this_run
            eta_min = (remaining / rate) / 60 if rate > 0 else float("nan")
            print(f"checkpoint: {len(done_links) + n_done_this_run}/{len(df)} total "
                  f"| this run {n_done_this_run}/{len(todo)} "
                  f"| {rate:.2f} items/s | ETA {eta_min:.1f} min")
            buffer = []

    if buffer:
        chunk_df = pd.DataFrame(buffer)
        write_header = not os.path.exists(OUT_PATH)
        chunk_df.to_csv(OUT_PATH, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    total_labeled = len(done_links) + n_done_this_run
    remaining_after = len(df) - total_labeled
    if time_limit_hit:
        print(f"time limit ({TIME_LIMIT_SECONDS}s) reached -- stopped for this run")
    print(f"labeled so far: {total_labeled}/{len(df)} (remaining: {remaining_after})")
    print("done")
