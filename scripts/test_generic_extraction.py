"""네이버가 아닌 언론사 링크에서 사이트별 규칙 없이 본문을 뽑을 수 있는지 시험한다.
CPU/네트워크만 사용. 설치 없이 bs4만 사용: (1) 흔한 본문 컨테이너 선택자, (2) 없으면 문단(<p>) 밀도가 가장 높은 영역.
정확성 검사: 추출한 본문에 네이버 API description(리드 스니펫)의 일부가 들어있는지 확인.
"""
import re
import sys
import time
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
IN_PATH = "data/processed/news_qwen_labeled.csv"
OUT_PATH = "data/raw/generic_extract_test.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SELECTORS = ["#article-view-content-div", "#articleBody", "#article-body", ".article-body",
             ".article_body", ".news_body", "#newsBody", "article"]
MIN_LEN = 300


def shape(link: str) -> str:
    p = urlparse(link)
    q = re.sub(r"=[^&]*", "", p.query)
    return re.sub(r"\d+", "N", p.path) + ("?" + q if q else "")


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
    df = df[df["qwen_label"].isin(["긍정", "부정", "중립"])]
    df = df[~df["link"].astype(str).str.contains("naver.com")]
    df = df[df["description"].notna()]
    sample = df.sample(n=N, random_state=42).reset_index(drop=True)

    rows = []
    for i, r in sample.iterrows():
        rec = {"link": r["link"], "domain": urlparse(r["link"]).netloc.replace("www.", ""),
               "shape": shape(r["link"]), "status": "", "method": "", "body_len": 0, "verified": None}
        try:
            resp = requests.get(r["link"], headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                rec["status"] = f"http_{resp.status_code}"
            else:
                if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
                    resp.encoding = resp.apparent_encoding
                method, body = extract(resp.text)
                rec["method"], rec["body_len"] = method, len(body)
                rec["verified"] = verified(body, r["description"])
                rec["status"] = "ok" if len(body) >= MIN_LEN else "short"
        except requests.exceptions.Timeout:
            rec["status"] = "timeout"
        except Exception as e:
            rec["status"] = "error:" + type(e).__name__
        rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{N}")
        time.sleep(0.3)

    out = pd.DataFrame(rows)
    out["group"] = out["shape"].map(lambda s: "articleView" if s.startswith("/news/articleView.html") else "other")
    out["good"] = (out["status"] == "ok") & (out["verified"] == True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== status ===")
    print(out["status"].value_counts().to_string())
    print("\n=== by group: n / extracted(ok) / good(ok+snippet verified) ===")
    g = out.groupby("group").agg(n=("link", "count"), ok=("status", lambda s: (s == "ok").sum()), good=("good", "sum"))
    g["good_rate"] = (g["good"] / g["n"] * 100).round(0)
    print(g.to_string())
    print("\n=== by method (ok rows only): n / good ===")
    m = out[out["status"] == "ok"].groupby("method").agg(n=("link", "count"), good=("good", "sum"))
    print(m.to_string())
    print(f"\nOVERALL good: {int(out['good'].sum())}/{len(out)} = {out['good'].mean()*100:.0f}%")
    print(f"domains in sample: {out['domain'].nunique()} | good domains: {out[out['good']]['domain'].nunique()}")
    print(f"saved to {OUT_PATH}")
