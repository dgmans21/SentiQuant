"""Conservative rule-based safety net: catch sports/esports GAME-RESULT
reporting that shares a name with the company (KT->kt wiz/kt Rolster,
기아->KIA 타이거즈/디플러스 기아, 삼성전자->삼성 라이온즈, 한화에어로스페이스
->한화 이글스 등) that Qwen mislabeled as 긍정/부정/중립 instead of 무관.

IMPORTANT distinction learned the hard way: many conglomerates genuinely
*sponsor* sports (KB금융의 골프대회, 셀트리온 퀸즈 마스터스 등) and news about
that sponsorship IS real business/PR content -- it should stay labeled, not
be treated as noise. Golf-sponsorship terms (KLPGA/PGA/버디/홀인원) were
tried and produced exactly this false-positive pattern, so they are
deliberately excluded here. Only pure GAME-RESULT vocabulary for baseball/
esports (which does not double as sponsorship-announcement language) is
used.

--dry-run flag: print candidates + which term matched, without writing.
"""
import sys
import pandas as pd

IN_PATH = "data/processed/news_qwen_labeled.csv"
OUT_PATH = "data/processed/news_qwen_labeled.csv"
WEAK_THRESHOLD = 2

# unambiguous game-result vocabulary -- a single hit is enough
STRONG_TERMS = [
    "이닝", "탈삼진", "방어율", "완봉", "완투", "세이브", "홀드", "볼넷", "사구",
    "피안타", "자책점", "출루율", "장타율", "병살타", "대타", "대주자",
    "선발투수", "구원투수", "퀄리티스타트", "평균자책점", "위즈", "라이온즈",
    "타이거즈", "이글스", "트윈스", "베어스", "자이언츠", "히어로즈", "랜더스",
    "와이번스", "KBO", "프로야구",
    "LCK", "롤챔스", "kt Rolster", "kt 롤스터", "디플러스", "젠지", "T1",
    "담원", "DRX", "리그오브레전드", "롤드컵", "밴픽",
]

# metaphor-prone in Korean financial headlines ("반도체 시장서 완승") --
# require 2+ hits so an isolated metaphor doesn't get caught
WEAK_TERMS = [
    "홈런", "안타", "타율", "다승", "타점", "득점권", "투수", "타자", "야구",
    "승리투수", "완봉승", "끝내기", "역전승", "선발승", "결승타", "역전홈런",
    "만루포", "만루홈런", "쐐기포", "실점",
]


def strong_hits(text: str) -> list:
    return [t for t in STRONG_TERMS if t in text]


def weak_hits(text: str) -> list:
    return [t for t in WEAK_TERMS if t in text]


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    df = pd.read_csv(IN_PATH)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["strong_hits"] = df["text"].apply(strong_hits)
    df["weak_hits"] = df["text"].apply(weak_hits)
    df["strong_n"] = df["strong_hits"].apply(len)
    df["weak_n"] = df["weak_hits"].apply(len)

    before = df["qwen_label"].value_counts()

    reclass_mask = (
        ((df["strong_n"] >= 1) | (df["weak_n"] >= WEAK_THRESHOLD))
        & (df["qwen_label"] != "무관")
    )
    print(f"재분류 대상: {reclass_mask.sum()}건")
    print(f"  - strong 매칭: {((df['strong_n'] >= 1) & (df['qwen_label'] != '무관')).sum()}건")
    print(f"  - weak만 {WEAK_THRESHOLD}개 이상: "
          f"{(((df['strong_n'] == 0) & (df['weak_n'] >= WEAK_THRESHOLD)) & (df['qwen_label'] != '무관')).sum()}건")

    print("\n=== 종목별 건수 ===")
    print(df.loc[reclass_mask, "stock"].value_counts().to_string())

    cand = df.loc[reclass_mask, ["stock", "qwen_label", "title", "strong_hits", "weak_hits"]]
    cand.to_csv("data/raw/reclass_candidates_v3.csv", index=False, encoding="utf-8-sig")
    print(f"\n전체 후보 목록 -> data/raw/reclass_candidates_v3.csv (검토용)")

    if dry_run:
        print("\n--dry-run 모드: 파일 저장 안 함")
        sys.exit(0)

    df.loc[reclass_mask, "qwen_label"] = "무관"
    df.drop(columns=["text", "strong_hits", "weak_hits", "strong_n", "weak_n"], inplace=True)

    after = df["qwen_label"].value_counts()
    print("\n=== 재분류 전 ===")
    print(before)
    print("\n=== 재분류 후 ===")
    print(after)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nsaved to {OUT_PATH}")
