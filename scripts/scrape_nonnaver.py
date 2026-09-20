"""비-naver 애매 케이스 전량을 bs4로 스크래핑. test_generic_extraction.py의 추출 로직 재사용.
설치 없음, CPU/네트워크만. 200건마다 체크포인트 저장, 링크 기준 재시작.
"""
import os
import re
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

IN_PATH = "data/raw/ambiguous_cases_nonnaver.csv"
OUT_PATH = "data/raw/nonnaver_scraped.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SELECTORS = ["#article-view-content-div", "#articleBody", "#article-body", ".article-body",
             ".article_body", ".news_body", "#newsBody", "article"]


def clean_text(el) -> str:
    for t in el.select("script, style, noscript, iframe"):
        t.decompose()
    return re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True))


def extract(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for sel in SELECTORS:
        el = soup.select_one(sel)
        if el is not None:
            text = clean_text(el)
            if len(text) >= 200:
                return "sel:" + sel, text
    scores = {}
    for p in soup.find_all("p"):
        n = len(p.get_text(strip=True))
        if n < 30:
            continue
        parent = p.parent
        if parent is not None:
            scores[parent] = scores.get(parent, 0) + n
            if parent.parent is not None:
                scores[parent.parent] = scores.get(parent.parent, 0) + n / 2
    if scores:
        best = max(scores, key=scores.get)
        return "fallback:p-density", clean_text(best)
    return "none", ""


def verified(body: str, desc: str):
    d = re.sub(r"\s+", "", str(desc)).replace("...", "").replace("…", "")
    chunks = [d[i:i + 16] for i in (0, 16, 32) if len(d) >= i + 16]
    if not chunks:
        return None
    b = re.sub(r"\s+", "", body)
    return any(c in b for c in chunks)


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

    done_links = set()
    if os.path.exists(OUT_PATH):
        done_links = set(pd.read_csv(OUT_PATH, encoding="utf-8-sig")["link"])
        print(f"resuming: {len(done_links)} rows already scraped")

    todo = df[~df["link"].isin(done_links)].reset_index(drop=True)
    print(f"remaining: {len(todo)} / {len(df)}")

    buffer = []
    for i, r in todo.iterrows():
        rec = dict(r)
        rec["full_body"] = ""
        rec["method"] = ""
        rec["good"] = False
        try:
            resp = requests.get(r["link"], headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
                    resp.encoding = resp.apparent_encoding
                method, body = extract(resp.text)
                rec["method"] = method
                if len(body) >= 300 and verified(body, r["description"]):
                    rec["full_body"] = body[:1500]
                    rec["good"] = True
        except Exception as e:
            rec["method"] = "error:" + type(e).__name__
        buffer.append(rec)

        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(todo)}")
        if (i + 1) % 200 == 0:
            chunk = pd.DataFrame(buffer)
            chunk.to_csv(OUT_PATH, mode="a", index=False, header=not os.path.exists(OUT_PATH), encoding="utf-8-sig")
            buffer = []
        time.sleep(0.3)

    if buffer:
        chunk = pd.DataFrame(buffer)
        chunk.to_csv(OUT_PATH, mode="a", index=False, header=not os.path.exists(OUT_PATH), encoding="utf-8-sig")

    out = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
    print(f"\n총 {len(out)}건, 성공(good) {int(out['good'].sum())}건 ({out['good'].mean()*100:.0f}%)")
    print(f"saved to {OUT_PATH}")
