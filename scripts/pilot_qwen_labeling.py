"""Pilot test: can Qwen3-4B label Korean financial news sentiment well enough
to use as a training-label source? Samples ~25 articles, asks Qwen for a
sentiment label + one-line reason, saves for manual eyeballing.
"""
import time
import pandas as pd
from unsloth import FastModel

N_SAMPLES = 25
DATA_PATH = "data/processed/news_labeled_excess_v2.csv"
OUT_PATH = "data/raw/pilot_qwen_labels.csv"

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


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(n=N_SAMPLES, random_state=7).reset_index(drop=True)

    t_load_start = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastModel.for_inference(model) if hasattr(FastModel, "for_inference") else model
    load_time = time.time() - t_load_start
    print(f"model load time: {load_time:.1f}s")

    results = []
    item_times = []
    for i, row in sample.iterrows():
        t0 = time.time()
        prompt = PROMPT_TEMPLATE.format(
            stock=row["stock"], title=row["title"], description=row.get("description", "")
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        ).to(model.device)

        output = model.generate(
            inputs, max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None
        )
        response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

        dt = time.time() - t0
        item_times.append(dt)

        results.append({
            "stock": row["stock"],
            "title": row["title"],
            "description": row.get("description", ""),
            "price_based_label": row["label"],
            "qwen_response": response.strip(),
        })
        print(f"[{i+1}/{N_SAMPLES}] {dt:.2f}s {row['stock']}: {response.strip()[:60]}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    avg_item = sum(item_times) / len(item_times)
    total_target = 11188
    est_seconds = avg_item * total_target
    print(f"\nmodel load time: {load_time:.1f}s")
    print(f"avg per-item time: {avg_item:.2f}s (n={len(item_times)})")
    print(f"estimated time for {total_target} items: {est_seconds/60:.1f} min ({est_seconds/3600:.2f} h)")
    print(f"\nsaved to {OUT_PATH}")
