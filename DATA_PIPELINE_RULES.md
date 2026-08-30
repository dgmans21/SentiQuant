# 데이터 파이프라인 운영 규칙

WORKLOG.md가 "무슨 일이 있었는지" 기록하는 일지라면, 이 문서는 "다음에 뭘 어떻게 해야 하는지" 확인하는 체크리스트/규칙집이다. 파이프라인을 다시 돌릴 때마다 이 문서부터 확인한다.

---

## 1. 파이프라인 순서 (의존관계)

```
1. 뉴스 수집
   - collect_news_backfill.py      → data/raw/news_backfill.csv (18종목, 1회성이었으나 주기 재실행 필요, 아래 2번 참고)
   - collect_news_curated.py       → data/raw/news_daily_curated.csv (삼성전자·SK하이닉스 전용, 매일)

2. 가격/지수 최신화 (매칭 전 항상 먼저)
   - collect_price_history.py      → data/raw/price_history.csv (종목별 종가, 1년치 통째로 재수집)
   - collect_index_history.py      → data/raw/index_history.csv (코스피, 1년치 통째로 재수집)

3. 뉴스-주가 매칭
   - match_news_price.py           → data/processed/news_price_matched.csv
     (뉴스 발행일 다음 거래일 종가 변동률. 다음 거래일 데이터가 없으면 "너무 최근"으로 skip됨 -- 2번을 먼저 갱신해야 풀림)

4. 초과수익률 계산 (KOSPI 대비)
   - add_excess_return.py          → news_price_matched.csv를 덮어씀 (index_return_pct, excess_return_pct 컬럼 추가)

5. Qwen3-4B 논조 라벨링
   - label_qwen_full.py            → data/processed/news_qwen_labeled.csv
     - DATA_PATH가 가리키는 입력 파일에 있는 것만 라벨링함. 3번 결과 전체가 입력에 들어있는지 반드시 확인 (아래 "체크리스트" 참고)
     - link 기준 체크포인트/재시작 가능, 목표건수 인자로 짧게 끊어서 실행 가능
     - 속도 실측: 배치(8) 기준 초당 1.3~1.5건

6. Train/val/test 분할
   - split_qwen_labeled.py         → data/processed/news_sentiment_final.csv
     - 무관/파싱실패 제외 후, (종목 x 라벨) 조합 기준 70/15/15 행 단위 stratified split
     - 예전엔 "날짜" 기준으로 쪼갰으나(주가 기반 라벨 시절 동일 날짜 시장 움직임 leakage 방지 목적), Qwen 라벨은 텍스트 자체 속성이라 그 leakage 우려가 없음 → 종목 단위 stratify로 전환 (2026-08-28). 반드시 이 방식 유지할 것 -- 날짜 기준으로 되돌리면 저빈도/짧은-span 종목이 test에서 통째로 빠지는 문제 재발함

7. 재학습
   - train_classifier.py <입력.csv> <모델출력경로>
     - 항상 klue/bert-base에서 새로 시작 (이전 체크포인트에 이어서 하지 않음)
     - split을 바꿨으면 반드시 처음부터 재학습해야 비교가 유효함
```

## 2. 종목별 수집 주기 티어 (2026-08-28 실측 기준)

백필은 NAVER API 결과 1000건 한도가 있어서, 하루 발행량이 많은 종목일수록 백필이 커버하는 기간이 짧아진다 (예: 삼성전자 하루 663건 → 3일치만 커버). 그래서 "며칠에 한 번 재수집"은 종목마다 다르게 가야 한다.

| 티어 | 주기 권장 | 종목 (2026-08-28 기준 백필 span) |
|---|---|---|
| 1 (고빈도, span ≤7일) | **매일** | 삼성전자(3일)·SK하이닉스(3일)·카카오(5일)·KT(6일)·현대차(6일)·삼성물산(6일)·SK텔레콤(7일)·기아(7일) |
| 2 (중빈도, span 8~16일) | **주 2회** (예: 월/목) | LG에너지솔루션·한화에어로스페이스·NAVER·KB금융·셀트리온·삼성바이오로직스·HD현대중공업·삼성SDI |
| 3 (저빈도, span 17일+) | **주 1회** | LG화학(24일)·LG생활건강(27일)·신한지주(29일)·POSCO홀딩스(120일) |

- 삼성전자·SK하이닉스는 `collect_news_curated.py`(POOL_SIZE=1000, TARGET_COUNT=300)로 이미 매일 루틴 존재.
- 나머지 티어 1 종목(카카오·KT·현대차·삼성물산·SK텔레콤·기아)은 **아직 매일 루틴이 없음** — `collect_news_backfill.py`를 정기 재실행하거나, curated 스크립트의 `DEFAULT_TARGETS`를 확장하는 방향 검토 필요.
- span 수치는 시간이 지나면 바뀔 수 있으므로(뉴스량 변동), 주기적으로 재확인할 것: `news_backfill.csv`에서 종목별 `news_date` unique 개수/min/max로 재계산 가능.

## 3. 재학습 전 체크리스트

1. **가격/지수 최신인가?** `price_history.csv`/`index_history.csv`의 최신 날짜 확인. 오래됐으면 1) 먼저 갱신.
2. **매칭 vs 라벨링 갯수가 같은가?** (2026-08-28에 이걸 놓쳐서 7,212건이 조용히 밀려 있었음)
   ```python
   matched = pd.read_csv('data/processed/news_price_matched.csv')['stock'].value_counts()
   labeled = pd.read_csv('data/processed/news_qwen_labeled.csv')['stock'].value_counts()
   gap = (matched - labeled).fillna(matched)  # 종목별로 0이어야 정상
   ```
   0이 아닌 종목이 있으면 5번(label_qwen_full.py) 전에 해당 분량을 입력 파일에 합쳐야 함.
3. **split이 종목 기준(stratified)인가?** `news_sentiment_final.csv`의 종목별 split crosstab에 0건 칸이 있으면 안 됨.
4. **표본 크기 해석 시 오차범위 감안**: test n=20이면 ±20%p대, n=100+이면 ±9%p대. 소표본 종목 개별 수치 차이에 과도한 의미 부여 금지.

## 4. 파일 버전 메모 (2026-08-28 기준)

- `data/processed/news_labeled_excess_v2.csv`, `v3.csv`, `v4.csv`: Qwen 라벨링 입력용으로 임시로 만든 중간 산출물. **v4가 가장 최신(전체 21,252건 포함)**이지만, 이후 `news_price_matched.csv`가 갱신되면 v4도 다시 stale해짐 -- 재사용하지 말고 그때그때 `news_price_matched.csv`에서 새로 만들 것.
- `models/klue-bert-qwen-sentiment` (구, 11,188건 학습) → `-v2` (12,548건, 종목 split 수정 직후) → `-v3` (18,196건, 전체 라벨링 완료 후, **현재 최종**).
