"""사람 검증용 블라인드 시트 생성: 재라벨링에서 라벨이 바뀐(flip) 기사 중 무작위 N건.
시트(md)에는 기사만 보여주고 이전/새 라벨은 별도 key 파일에 숨긴다.
사용법: make_human_check_sheet.py N [START_ID SHEET_PATH]
  START_ID > 1 이면 기존 key 파일에 이어 붙이고(이미 뽑힌 기사는 제외), 시트는 SHEET_PATH에 새로 만든다.
"""
import os
import re
import sys
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
START_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SHEET_PATH = sys.argv[3] if len(sys.argv) > 3 else "data/raw/human_check_sheet.md"
SEED = 7 if START_ID == 1 else 8
BODY_CHARS = 1000
KEY_PATH = "data/raw/human_check_key.csv"
VALID = {"긍정", "부정", "중립", "무관"}

nv = pd.read_csv("data/raw/full_body_relabeled_full.csv", encoding="utf-8-sig")
nv = nv.merge(pd.read_csv("data/raw/full_body_scraped_full.csv", encoding="utf-8-sig")[["link", "full_body"]],
              on="link", how="left").assign(source="naver")
nn = pd.read_csv("data/raw/nonnaver_relabeled.csv", encoding="utf-8-sig")
nn = nn.merge(pd.read_csv("data/raw/nonnaver_scraped.csv", encoding="utf-8-sig")[["link", "full_body"]],
              on="link", how="left").assign(source="nonnaver")
df = pd.concat([nv, nn], ignore_index=True)
df = df[(df["snippet_label"] != df["full_body_label"]) & df["full_body_label"].isin(VALID)
        & df["full_body"].notna() & (df["full_body"].astype(str).str.len() > 200)]

if START_ID > 1:
    used = set(pd.read_csv(KEY_PATH, encoding="utf-8-sig")["link"])
    df = df[~df["link"].isin(used)]
print(f"flip candidates: {len(df)} (naver {int((df['source']=='naver').sum())}, nonnaver {int((df['source']=='nonnaver').sum())})")

s = df.sample(n=N, random_state=SEED).reset_index(drop=True)
s["id"] = s.index + START_ID


def one_line(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


if START_ID == 1:
    header = """# 사람 검증 시트 (블라인드)

**할 일**: 아래 기사마다 맨 아래 `판단:` 뒤에 **한 글자**만 적어주세요.

- `긍` — 이 기사가 해당 **종목**에 대해 전달하는 논조가 긍정적
- `부` — 부정적 (실적 부진, 하락, 리스크 등)
- `중` — 중립적 (객관적 사실 전달, 방향성 없음)
- `무` — 종목과 **무관**한 기사 (이름만 우연히 겹치거나 다른 회사 소식에 곁가지로 등장)

기준: "객관적 사실 전달이라도 내용이 실적 부진·하락·리스크를 담으면 부정, 성장·호실적·상승을 담으면 긍정"으로 봅니다(Qwen에게 준 기준과 같음). 본문은 앞부분 1,000자까지만 보여줍니다.
**틀려도 괜찮습니다.** 애널리스트처럼 판단하려 하지 말고, 기사가 그 종목을 어떻게 다루는지(좋은 소식으로 쓰는지, 나쁜 소식으로 쓰는지)를 보고 15~20초 안에 첫인상으로 고르세요. 기사만 봐선 좋은 소식인지 나쁜 소식인지 방향이 안 보이면(예: 인수 시도, 지분 변동) `중`을 적거나, 정말 모르겠으면 `?`를 적으세요.
**정답 파일(`human_check_key.csv`)은 채점이 끝날 때까지 열지 마세요.** 다 하셨거나 중간까지만 하셨으면 알려주세요.

---
"""
else:
    header = f"""# 사람 검증 시트 2 ({START_ID}~{START_ID + N - 1}번, 블라인드)

기준과 방법은 1번 시트(`human_check_sheet.md`)와 같습니다. 기사마다 `판단:` 뒤에 `긍`/`부`/`중`/`무`/`?` 중 한 글자를 적어주세요. 정답 파일은 채점 전까지 열지 마세요.

---
"""
parts = [header]
for _, r in s.iterrows():
    body = one_line(r["full_body"])[:BODY_CHARS]
    parts.append(f"## {r['id']}\n\n- 종목: {r['stock']}\n- 제목: {one_line(r['title'])}\n- 본문: {body}\n\n**판단:** \n\n---\n")
open(SHEET_PATH, "w", encoding="utf-8", newline="\n").write("\n".join(parts))

key = s[["id", "source", "stock", "link", "snippet_label", "full_body_label"]].rename(
    columns={"snippet_label": "old_label", "full_body_label": "new_label"})
key.to_csv(KEY_PATH, mode="a" if START_ID > 1 else "w", header=(START_ID == 1), index=False, encoding="utf-8-sig")
print(f"wrote {SHEET_PATH} and {'appended to' if START_ID > 1 else 'wrote'} {KEY_PATH} ({N} items, ids {START_ID}-{START_ID + N - 1})")
print("source mix:", s["source"].value_counts().to_dict())
