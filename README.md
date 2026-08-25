# SentiQuant

AI 뉴스 기반 한국 주식 감성분석 프로젝트. 상세 배경/단계별 계획은 [project_roadmap.md](project_roadmap.md), 날짜별 상세 작업 기록은 [WORKLOG.md](WORKLOG.md) 참고. 이 문서는 **현재 상태 스냅샷**이다 — 새 세션(사람이든 AI든)이 여기서부터 이어갈 수 있도록 지금 무엇이 되어있고 무엇이 남았는지만 정리한다.

## 환경

- Conda 환경: `unsloth_env` (Python 3.12) — `conda activate unsloth_env`
- PyTorch 2.11.0+cu130, GPU: RTX 4060 Ti 8GB (로컬)
- 핵심 패키지: unsloth, transformers, peft, trl, bitsandbytes, scikit-learn
- API 키: `.env` 파일에 `NAVER_API_KEY_ID`, `NAVER_API_KEY` (NAVER Cloud Platform "NAVER API HUB" 뉴스검색). 발급 관련 함정은 WORKLOG 2026-08-23 참고.

## 지금 상태 (2026-08-26 기준)

**핵심 설계 결정 (가장 중요)**: 처음엔 "뉴스 → 다음날 주가 등락(상승/하락/중립)"을 라벨로 학습시켰으나 성능이 찍기 수준(~33%)을 못 넘김. **"뉴스만으로 익일 주가를 예측하는 건 과도한 목표"라고 판단하고 방향 전환**: 라벨을 주가 변동이 아니라 **기사 자체의 논조(긍정/부정/중립/무관)**로 바꾸고, 주가는 학습 후 "논조가 실제 주가와 상관있는지" 검증하는 별도 단계로 이동시킴. 자세한 실험 결과와 이유는 WORKLOG 2026-08-26 참고.

### 데이터 파이프라인 (완료)

1. **수집**: 20종목(섹터 분산) × NAVER 뉴스검색 API + yfinance 주가 → `data/raw/news_backfill.csv` (21,286건), `data/raw/price_history.csv`, `data/raw/index_history.csv`(코스피 지수)
   - 삼성전자·SK하이닉스는 뉴스량이 너무 많아 API 1000건 한도가 반나절 만에 소진됨 → 별도로 하루 80건씩 큐레이션(`collect_news_curated.py`)해서 누적 중. **사용자가 매일 직접 수동 실행**해야 함 (자동화 안 됨)
2. **매칭**: 뉴스 발행일 → 다음 거래일 종가 변동률 계산 → `data/processed/news_price_matched.csv` (11,188건, `return_pct`/`excess_return_pct` 컬럼 포함)
3. **논조 라벨링 (진행 중)**: Qwen3-4B(로컬, unsloth)로 각 기사의 긍정/부정/중립/**무관** 판정 → `data/processed/news_qwen_labeled.csv`
   - **진행률: 4,000 / 11,188건 (35.8%)** — 이어서 하려면 `python scripts\label_qwen_full.py <목표건수>` (링크 기준 체크포인트, 언제 끊어도 안전)
   - 지금까지 분포: 중립 41% / **무관 23%** / 긍정 19% / 부정 17% (파싱 실패 0)

### 모델 (실험 단계, 전부 폐기 예정 — 참고용으로만 보존)

`models/` 아래 4개 체크포인트는 전부 **"주가 기반 라벨"로 학습한 실패작들**이다 (git에는 안 올라감, 로컬에만 있음):
- `klue-bert-sentiment`, `klue-bert-return`, `klue-bert-excess`, `klue-bert-excess-v2` — 최고 성능이 accuracy 38.5%(찍기 33.3%)에 그침
- **다음 재학습은 이것들을 대체한다.** Qwen 라벨링이 끝나면 "무관" 제외 후 긍정/부정/중립 3분류로 KLUE-BERT를 새로 학습할 예정 (`scripts/train_classifier.py` 재사용, 데이터 경로만 `news_qwen_labeled.csv` 기반으로 바꾸면 됨 — 아직 이 연결 스크립트는 안 만들어짐)

## 다음에 할 일 (우선순위 순)

1. **Qwen 라벨링 완료** (남은 7,188건) — `python scripts\label_qwen_full.py <목표건수>`
2. `news_qwen_labeled.csv`에서 "무관" 제외 → train/val/test 분할 (날짜 그룹 + 랜덤 셔플, `label_and_split.py` 패턴 재사용) → KLUE-BERT 재학습
3. 재학습된 모델로 "논조 예측 vs 실제 주가" 상관관계 사후 검증 분석 설계
4. (병행, 느긋하게) 삼성전자·SK하이닉스 일일 큐레이션 계속, 매칭 보류분 주기적 재확인

## 스크립트 맵

| 스크립트 | 역할 |
|---|---|
| `collect_multi_sample.py` | 20종목 리스트(`STOCKS` dict) 정의 + 소량 테스트 수집 |
| `collect_news_backfill.py` | 종목당 최대 1000건 뉴스 백필 (재실행 시 기존과 merge+dedup) |
| `collect_news_curated.py` | 삼성전자·SK하이닉스 전용 일일 소량(80건) 큐레이션 수집 |
| `collect_price_history.py` / `collect_index_history.py` | 종목/코스피 지수 1년치 주가 |
| `match_news_price.py` | 뉴스+주가 매칭 (backfill + curated 자동 병합) |
| `add_excess_return.py` | 코스피 대비 초과수익률 계산 (현재는 미사용 방향) |
| `label_and_split.py` | 임계값 기반 라벨링 + 그룹셔플 split (주가 기반, 현재는 미사용 방향) |
| `find_good_seed.py` | split 시드 탐색 (라벨 분포 균형용) |
| `train_classifier.py` | KLUE-BERT 파인튜닝 (데이터 경로/출력 경로 인자로 받음) |
| `pilot_qwen_labeling.py` | Qwen3 라벨링 소규모 파일럿 테스트용 |
| `label_qwen_full.py` | Qwen3 전체 데이터 라벨링 (배치+체크포인트, 진행 중인 핵심 스크립트) |
| `check_*.py` | 각종 QA/진단 유틸리티 (분포 확인, 타임스탬프 갭 확인 등) |
