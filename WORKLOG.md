# 작업 일지

새 작업일마다 최상단에 날짜 섹션을 추가한다 (최신순).

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
