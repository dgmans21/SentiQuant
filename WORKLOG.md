# 작업 일지

새 작업일마다 최상단에 날짜 섹션을 추가한다 (최신순).

---

## 2026-08-26

### Phase 3 — KLUE-BERT 파인튜닝 1차 시도 (주가 기반 라벨)
- 8,140건(원래 수익률, ±1.0% 임계값)으로 학습 → **test accuracy 25.7~31.7%, F1(macro) 25.8~30.3%** (찍기 확률 33.3%보다 낮거나 비슷)
- 혼동행렬 확인 결과 모델이 특정 클래스(상승 또는 중립)로 쏠려 도망가는 패턴, 실질적 신호 학습 실패
- **원인 진단**: (1) 헤드라인 텍스트만으로 익일 주가 방향을 예측하는 과제 자체가 원래 어려움(효율적 시장 가설과 관련), (2) 시장 전체가 오르내리는 날엔 무관한 기사도 다 같은 라벨이 붙는 라벨 노이즈

### 개선 시도 1 — 시장 대비 초과수익률 라벨링
- 코스피 지수(^KS11) 수익률을 종목 수익률에서 빼서 "초과수익률" 계산 (`scripts/add_excess_return.py`)
- 재학습 결과: accuracy 35.0%, F1 32.2% — 찍기 확률 첫 돌파, 다만 혼동행렬 보면 "중립"으로 쏠리는 패턴 여전
- **Split 시행착오 추가 발견**: 날짜가 111개뿐이라 무작위 셔플이라도 시드에 따라 라벨 분포가 크게 출렁임 → 여러 시드 탐색 후 최적 시드 채택 (`scripts/find_good_seed.py`)

### 개선 시도 2 — 데이터 확장
- 18개 종목 백필 재실행(15,877→21,286건), 주가/지수 데이터 갱신, 보류됐던 매칭 재시도, 삼성전자·SK하이닉스 큐레이션 데이터(112건) 매칭 파이프라인에 연결
- 매칭 결과 8,140→11,188건으로 확장
- 재학습 결과: **accuracy 38.5%, F1(macro) 38.5%** — 지금까지 최고 성능. 혼동행렬도 한 클래스로 쏠리지 않고 세 방향 다 어느 정도 구분 (하락 회수율만 약함, 24%)

### 방향 전환 논의 — "뉴스만으로 주가 예측"의 한계 인정
- 사용자가 "기사만으로 주가 예측은 과도한 기대 아니냐"는 의문 제기 → 타당한 지적으로 판단
- 대안: 라벨을 "주가 변동"이 아니라 "기사 자체의 논조(긍정/부정/중립)"로 바꾸고, 주가는 학습 후 별도 검증 단계로 이동
- 논조 라벨은 Qwen3-4B로 자동 생성하기로 결정 (사람이 대량 수작업 라벨링 어려움)

### Qwen3-4B 자동 라벨링 파일럿
- 1차 테스트: `max_new_tokens=80`이 너무 짧아 Qwen3의 "생각 모드"(`<think>`)에 응답이 전부 잘림 → `enable_thinking=False`로 해결
- 2차 테스트(25건, 3분류): 대체로 합리적이나 **무관 기사(예: "KT" 검색에 kt wiz 야구팀 기사가 걸림)를 못 거름**
- 3차 테스트(25건, 4분류로 "무관" 옵션 추가): 무관 기사 5/5 정확히 걸러냄, 노이즈 필터링 문제 해결 확인
- 속도: 건당 3.7초(비배치) → 배치 처리(`BATCH_SIZE=8`)로 초당 1.3~1.5건까지 개선 (약 5배)
- 전체 11,188건 예상 시간 약 2시간 → 10분/지정 건수 단위로 끊어서 체크포인트 저장하며 진행하는 방식 채택 (`scripts/label_qwen_full.py`, 링크 기준 재시작 가능)
- 오늘 최종 4,000/11,188건(35.8%) 완료 (10분 타임박스 1회 + 목표건수 지정 3회, 총 4번 실행)
  - 라벨 분포(3,000건 시점): 중립 41.2% / **무관 22.7%** / 긍정 19.3% / 부정 16.8%, 파싱 실패 0건
  - **무관 22.7%라는 수치로, 기존에 우려했던 키워드 검색 노이즈 문제가 실제로 상당했음을 정량적으로 확인**
- 실행 시간 실측(스크립트 자체 기준, 모델 로딩 포함): 880건/10.2분, 1,120건/14.6분, 1,000건/12.4분, 1,000건/약 12~13분 — 평균 초당 1.3건 내외로 안정적
- 사용자가 컴퓨터를 켜놓고 자고 싶지 않다고 해서, 매번 목표 건수를 지정해 짧게 끊어 실행하고 다음 진행 여부를 묻는 방식으로 운영 중 (체크포인트가 링크 기준이라 언제 끊어도 안전)

### 다음 할 일
- [ ] Qwen 라벨링 이어서 진행 (남은 7,188건) — `python scripts\label_qwen_full.py <목표건수>`
- [ ] 전체 완료 후: "무관" 제외하고 긍정/부정/중립 3분류로 KLUE-BERT 재학습, 성능 비교
- [ ] 재학습 후 "주가 검증 단계" 설계: Qwen이 긍정 판정한 기사들이 실제로 주가와 상관있는지 사후 분석
- [ ] (병행) 삼성전자·SK하이닉스 일일 큐레이션 계속, 보류 매칭 주기적 재시도

### 산출물
- `scripts/add_excess_return.py`, `scripts/find_good_seed.py`, `scripts/train_classifier.py`(경로 인자화)
- `scripts/pilot_qwen_labeling.py`, `scripts/label_qwen_full.py`, `scripts/check_qwen_progress.py`
- `data/processed/news_labeled_excess_v2.csv` (11,188건, 주가 기반 라벨)
- `data/processed/news_qwen_labeled.csv` (진행 중, 3,000/11,188건 논조 라벨)
- `models/klue-bert-return`, `models/klue-bert-excess`, `models/klue-bert-excess-v2` (실험별 체크포인트, git 미포함)

## 2026-08-25

### Phase 1 마무리 — 백필 + 뉴스-주가 매칭
- 20종목 각각 최대 1000건(API 한도) 백필 시도 → 첫 실행에서 종목별 편차 발견 (453~998건). 재조사 결과 진짜 원인은 **페이지네이션 드리프트**: 여러 페이지를 순차 호출하는 동안 실시간으로 새 기사가 계속 올라와서, 날짜순 정렬 결과가 호출 사이에 밀리며 중복/누락이 생김. 버그도 콘텐츠 한계도 아니라 자연스러운 현상으로 결론, 재시도 로직 추가 후 최종 15,877건(dedup) 확보
- 주가 데이터를 1개월 → **1년치로 재수집** (뉴스가 4월 말까지 있어서 커버 필요)
- 뉴스-주가 매칭 로직 구현: 뉴스 발행일 → **다음 거래일** 종가 변동률(전일 대비 %) 계산 (`scripts/match_news_price.py`)
  - 8,140건 매칭 성공, 7,737건은 최근(08-21~08-25) 뉴스라 다음 거래일 주가가 아직 데이터 소스에 안 올라와서(NaN) 매칭 보류 — 원본은 보존되며 며칠 뒤 재실행하면 자동 회수됨
  - 검증: 수동 계산(LG에너지솔루션 08-20→08-21 -4.05%)과 대조해서 로직 정확성 확인
- 삼성전자·SK하이닉스처럼 뉴스량이 너무 많아 1000건이 반나절 만에 소진되는 종목 대응: 하루 소량(80건)씩 제목 유사도 기반 중복 제거 + 언론사 쏠림 방지하며 큐레이션하는 `scripts/collect_news_curated.py` 제작 (매일 수동 실행 예정, 사용자 담당)

### Phase 2 — 라벨링 + Split
- 임계값 실험: 데이터 실제 분포(평균 +0.28%, 표준편차 3.29%) 기준으로 여러 후보 비교 → **±1.0%**가 우연히도 하락/중립/상승을 32.6/33.6/33.8%로 거의 완벽하게 3등분함, 이걸로 채택
- Train/val/test 분할에서 시행착오:
  1. 날짜 개수 기준 70/15/15 → 최근 날짜에 뉴스량이 쏠려서 실제 건수는 test 83% vs train 9%로 뒤집힘
  2. 건수 기준 블록 분할로 수정 → 블록 생성 로직 버그로 크기 왜곡
  3. 날짜 단위 7일 주기 라운드로빈으로 단순화 → **요일 효과**로 추정되는 라벨 쏠림 재발견 (test 하락 72%, val 상승 63%)
  4. 최종: 날짜를 고정 시드로 **랜덤 셔플 후 70/15/15 분할** (그룹 단위는 날짜 유지 → 같은 날 중복 기사가 train/test에 동시에 들어가는 누출은 방지) → train 5,457 / val 2,008 / test 675, 세 split 모두 합리적으로 균형
- 결과: `data/processed/news_labeled.csv` (label, split 컬럼 포함)

### 다음 할 일
- [ ] EDA (라벨 분포, 종목별/섹터별 시각화) — 선택 사항, Phase 3 진행에 필수는 아님
- [ ] 며칠 후: `collect_price_history.py` → `match_news_price.py` 재실행해서 보류된 7,737건 회수
- [ ] 삼성전자·SK하이닉스: `collect_news_curated.py` 매일 실행해서 누적
- [ ] Phase 3: KLUE-BERT/FinBERT-kr 파인튜닝 (예상: GPU 학습 자체 10~30분, 전체 작업 1~3시간)

### 산출물
- `scripts/collect_price_history.py`, `scripts/match_news_price.py`, `scripts/collect_news_curated.py`, `scripts/label_and_split.py`
- `scripts/check_backfill_range.py`, `scripts/check_match_gap.py`, `scripts/check_matched_stats.py`, `scripts/check_thresholds.py` (QA용 유틸리티)
- `data/raw/news_backfill.csv` (15,877건), `data/raw/price_history.csv` (1년치)
- `data/processed/news_price_matched.csv` (8,140건), `data/processed/news_labeled.csv` (최종 라벨+split)

---

## 2026-08-23

### Phase 0 — 환경 세팅 완료
- Conda 환경 `unsloth_env` (Python 3.12.13) 생성
- PyTorch 2.11.0+cu130 설치, RTX 4060 Ti GPU 인식 확인
- unsloth 2026.8.19 + transformers/peft/trl/bitsandbytes/accelerate/datasets 설치
- Qwen3-4B 4bit 로드 테스트 성공 (`scripts/test_qwen_load.py`)
- **이슈**: `pip install unsloth`가 index-url 없이 실행되면서 GPU용 torch를 CPU 빌드로 덮어씀 → cu130 인덱스로 강제 재설치해서 해결. 이후 패키지 추가 설치 시 GPU 인식 재확인 필요.

### Phase 1 — 데이터 수집 맛보기 테스트
- yfinance로 주가 수집 파이프라인 검증 (문제없음)
- 뉴스 수집 소스 탐색:
  - 네이버 금융 종목뉴스 위젯(`finance.naver.com/item/news_news.naver`) — 개편으로 더 이상 작동 안 함
  - Google News RSS — 작동은 하나 노이즈(무관 언론사, 리다이렉트 링크) 있음
  - **네이버 뉴스검색 API** — developers.naver.com에서는 신규 등록 불가 (`"신규로 등록할 수 없는 API가 선택되었습니다"` 에러). 확인 결과 검색 API가 **네이버클라우드플랫폼(NCP)의 "NAVER API HUB"로 이전**됨. `console.ncloud.com`에서 신청, 엔드포인트 `https://naverapihub.apigw.ntruss.com/search/v1/news`, 헤더 `X-NCP-APIGW-API-KEY-ID` / `X-NCP-APIGW-API-KEY`로 인증 방식 변경됨
- 최종적으로 네이버 뉴스검색 API 연동 성공, `.env`에 키 저장 (`NAVER_API_KEY_ID`, `NAVER_API_KEY`)
- 20종목(7개 섹터 분산: 반도체/2차전지/자동차/플랫폼/바이오/금융/화학에너지/철강조선/통신/유통소비재)으로 뉴스 400건 + 주가 23일치 수집 테스트 완료
- **알려진 이슈**: 종목명 단순 키워드 검색이라 무관 기사(연예 뉴스 등)가 섞임 → Phase 2 라벨링 전 필터링 필요

### 다음 할 일
- [ ] Phase 1 본격화: 수집 기간 6개월~1년으로 확장, 종목당 뉴스량 확대
- [ ] 뉴스 노이즈 필터링 (경제/증권 섹션 위주, 키워드 동시 등장 조건)
- [ ] 뉴스-주가 매칭 로직 설계 (발행일 → 다음 거래일 종가 변동률)
- [ ] Phase 2: 라벨링 규칙(상승/하락/중립 임계값) 확정, EDA

### 산출물
- `scripts/collect_sample.py`, `scripts/collect_multi_sample.py`, `scripts/test_naver_news_api.py`, `scripts/test_qwen_load.py`
- `.env` / `.env.example` (API 키 관리)
- `data/raw/*.csv` (테스트 수집 결과, git 미포함)
