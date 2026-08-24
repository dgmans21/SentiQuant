# 작업 일지

새 작업일마다 최상단에 날짜 섹션을 추가한다 (최신순).

---

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
