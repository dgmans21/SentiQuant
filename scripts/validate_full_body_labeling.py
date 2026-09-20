"""라벨 품질 검증: 애매한 케이스(중립<->긍정 혼동) 150건을 본문 스크래핑해서
요약문 기반 Qwen 라벨과 본문 전체 기반 Qwen 라벨을 비교, flip rate를 측정한다.
naver.com 뉴스 페이지 전용 스크래퍼(article body: #dic_area).
"""
import re
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ambiguous_cases.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/raw/full_body_scraped.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def scrape_naver_body(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        body = soup.select_one("#dic_area")
        if body is None:
            return ""
        for tag in body.select("script, style, span.end_photo_org"):
            tag.decompose()
        text = body.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text
    except Exception as e:
        return f"__ERROR__:{e}"


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    bodies = []
    for i, row in df.iterrows():
        body = scrape_naver_body(row["link"])
        bodies.append(body)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(df)}")
        if (i + 1) % 200 == 0:
            tmp = df.iloc[:i+1].copy()
            tmp["full_body"] = bodies
            tmp.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        time.sleep(0.2)

    df["full_body"] = bodies
    ok = df[~df["full_body"].str.startswith("__ERROR__", na=False)]
    empty = df[df["full_body"] == ""]
    err = df[df["full_body"].str.startswith("__ERROR__", na=False)]
    print(f"\n성공: {len(ok) - len(empty)}, 본문없음: {len(empty)}, 에러: {len(err)}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved to {OUT_PATH}")
