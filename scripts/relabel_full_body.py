"""애매한 케이스(naver 링크, 중립<->긍정/부정 혼동)를 본문 전체로 Qwen 재라벨링.
label_qwen_full.py와 동일한 체크포인트/재시작 패턴 -- 목표 건수 지정, 링크 기준 재시작.
"""
import os
import re
import sys
import pandas as pd
import torch
from unsloth import FastModel

IN_PATH = "data/raw/full_body_scraped_full.csv"
OUT_PATH = "data/raw/full_body_relabeled_full.csv"
MAX_ITEMS_THIS_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 150
MAX_BODY_CHARS = 1500
BATCH_SIZE = 8

PROMPT_TEMPLATE = """다음은 한국 주식 종목 관련 뉴스 제목과 본문이다. 아래 순서로 판단하라.

1단계: 이 기사가 실제로 "{stock}"이라는 기업/종목 자체의 사업, 실적, 주가, 경영 활동과 관련이 있는가?
   - 종목명이 우연히 언급될 뿐 실제 내용은 무관한 주제(예: 같은 이름의 스포츠팀, 다른 기업 소식에 곁가지로 등장, 오탐)라면 "무관"으로 답하고 종료하라.
2단계: 관련이 있다면, 이 기사가 "{stock}"에 대해 전달하는 논조가 긍정적인지 부정적인지 중립적인지 판단하라.
   - 객관적 사실 전달이라도 내용이 실적 부진, 하락, 리스크 등을 담고 있으면 부정으로, 성장, 호실적, 상승 등을 담고 있으면 긍정으로 판단하라.

종목: {stock}
제목: {title}
본문: {body}

아래 형식으로만 답하라:
라벨: <긍정|부정|중립|무관>
이유: <한 줄 이유>"""

LABEL_RE = re.compile(r"라벨:\s*(긍정|부정|중립|무관)")


def extract_label(text: str) -> str:
    m = LABEL_RE.search(text)
    return m.group(1) if m else "파싱실패"


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = df[df["full_body"].notna() & ~df["full_body"].astype(str).str.startswith("__ERROR__")]
    df = df[df["full_body"].astype(str).str.len() > 0].reset_index(drop=True)

    done_links = set()
    if os.path.exists(OUT_PATH):
        done_links = set(pd.read_csv(OUT_PATH)["link"])
        print(f"resuming: {len(done_links)} rows already relabeled")

    todo = df[~df["link"].isin(done_links)].reset_index(drop=True)
    print(f"remaining: {len(todo)} / {len(df)}")
    todo = todo.iloc[:MAX_ITEMS_THIS_RUN]

    if len(todo) == 0:
        print("done")
        sys.exit(0)

    import time
    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-4B", max_seq_length=2048, load_in_4bit=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t1 = time.time()
    print(f"모델 로딩: {t1-t0:.1f}초")

    buffer = []
    all_flips = []
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[i:i+BATCH_SIZE]
        prompts = [
            PROMPT_TEMPLATE.format(
                stock=r["stock"], title=r["title"],
                body=str(r["full_body"])[:MAX_BODY_CHARS]
            ) for _, r in batch.iterrows()
        ]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                enable_thinking=False, tokenize=False,
            ) for p in prompts
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
                "stock": row["stock"], "sector": row["sector"], "news_date": row["news_date"],
                "title": row["title"], "description": row["description"], "link": row["link"],
                "snippet_label": row["label"], "pred_label": row["pred_label"],
                "full_body_label": extract_label(resp),
                "full_body_response": resp.strip().replace("\n", " "),
            })
        print(f"{min(i+BATCH_SIZE, len(todo))}/{len(todo)} | {time.time()-t1:.1f}s elapsed")

        if len(buffer) >= 40:
            chunk_df = pd.DataFrame(buffer)
            write_header = not os.path.exists(OUT_PATH)
            chunk_df.to_csv(OUT_PATH, mode="a", index=False, header=write_header, encoding="utf-8-sig")
            all_flips.append((chunk_df["snippet_label"] != chunk_df["full_body_label"]).sum())
            buffer = []

    t2 = time.time()
    if buffer:
        chunk_df = pd.DataFrame(buffer)
        write_header = not os.path.exists(OUT_PATH)
        chunk_df.to_csv(OUT_PATH, mode="a", index=False, header=write_header, encoding="utf-8-sig")
        all_flips.append((chunk_df["snippet_label"] != chunk_df["full_body_label"]).sum())

    total_done = len(done_links) + len(todo)
    flipped = sum(all_flips)
    print(f"\n생성 소요: {t2-t1:.1f}초 ({len(todo)}건, {len(todo)/(t2-t1):.2f}건/초)")
    print(f"이번 배치 flip: {flipped}/{len(todo)}")
    print(f"전체 진행: {total_done}/{len(df)}")
    print("done")
