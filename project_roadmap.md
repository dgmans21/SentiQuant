# AI 뉴스 + 주가 영향 분석 프로젝트 로드맵

**기간**: 1~2주
**하드웨어**: RTX 4060 Ti 8GB (로컬 우선) + Kaggle Notebooks (백업)
**핵심 모델**: KLUE-BERT/FinBERT-kr(분류) + Qwen3-4B LoRA/QLoRA via Unsloth(요약/설명 생성)

---

## 우선순위 원칙

1. 핵심 파이프라인(데이터 → 분류 모델 → 평가)을 가장 먼저 완성한다. 이게 안 되면 나머지는 의미 없음.
2. LLM 파인튜닝(Qwen3-4B)은 두 번째 우선순위. 시간이 부족해지면 데이터셋을 더 줄여서라도 "완성된 작은 버전"을 남긴다.
3. Ollama/Open WebUI, NestJS/Spring 레이어는 시간이 남을 때 추가하는 확장 요소. 없어도 핵심 스토리는 성립한다.
4. 매 학습 단계마다 `save_steps` 체크포인트를 켜둔다 (로컬이라도 크래시/정전 대비).

---

## Phase 0 — 환경 세팅 (0.5일)

- Python venv + PyTorch, transformers, peft, trl, unsloth, bitsandbytes 설치
- Qwen3-4B 다운로드 및 4060 Ti에서 로드 테스트 (Unsloth `FastModel.from_pretrained`, 4bit)
- Colab에서 짧은 스모크 테스트로 학습 코드 자체가 에러 없이 도는지 먼저 검증 (본 학습은 로컬에서)

## Phase 1 — 데이터 수집 (1~1.5일)

- 뉴스: 뉴스 API 또는 크롤링으로 종목/시장 관련 뉴스 수집
- 주가: yfinance 등으로 동일 기간 주가 데이터 수집
- 뉴스 발행 시점 기준으로 이후 주가 변동(예: 익일 종가 변화율)과 매칭

## Phase 2 — 데이터셋 구축/라벨링 (1일)

- 라벨 정의: 상승/하락/중립 (또는 감성 점수) — 임계값 기반 자동 라벨링 + 샘플 수동 검증
- 학습/검증/평가 세트 분할
- sklearn/pandas로 기초 EDA, seaborn으로 클래스 분포·상관관계 시각화 (여기서 기존 스택 자연스럽게 활용)

## Phase 3 — 분류 모델 파인튜닝 (1~1.5일) — 핵심 지표

- KLUE-BERT 또는 FinBERT-kr을 PyTorch/TensorFlow로 파인튜닝 (뉴스 → 상승/하락/중립 분류)
- 가볍고 빠름 (로컬 4060 Ti로 충분, 몇 시간 내 완료)
- 평가: Accuracy/F1, 혼동행렬 시각화
- **이 단계가 끝나면 이미 "완성된 프로젝트"의 최소 형태가 확보됨**

## Phase 4 — LLM 파인튜닝: Qwen3-4B + LoRA/QLoRA (2~3일)

- Unsloth로 4bit QLoRA 세팅 (rank 16~32, batch size 1~2 + gradient accumulation, PagedAdamW)
- 태스크: 뉴스 요약 + 영향 근거 설명 생성 (분류와 역할 분리)
- 로컬 우선 진행, 세션 불안 시 Kaggle(주 30시간 GPU 쿼터)로 이동
- `save_steps`로 체크포인트 주기적 저장 → Hub나 로컬에 백업
- 데이터셋은 500~2000건 정도로 소규모 유지 (시간 절약)

## Phase 5 — 평가 및 경량화 배포 (1일)

- 파인튜닝 전/후 출력 비교표 작성 (README용 근거 자료)
- GGUF Q4_K_M로 export → Ollama 로컬 서빙 테스트
- Open WebUI 연결해서 모델 응답 검증용 데모로 세팅 (서비스 프론트와 역할 분리)

## Phase 6 — 백엔드 (1.5~2일)

- FastAPI: 분류 모델 + Qwen3 서빙 API
- PostgreSQL: 뉴스, 감성 라벨, 주가 시계열 테이블 설계 및 적재
- NestJS/Spring: 뉴스 수집 스케줄링, 인증, FastAPI 호출 오케스트레이션

## Phase 7 — React 대시보드 (2일)

- 감성 트렌드 × 주가 오버레이 시각화
- 필터/기간 선택 UI
- (시간 되면) pgvector로 유사 과거 뉴스 검색 기능 추가

## Phase 8 — 통합 테스트 + 버퍼 (1일)

- 전체 파이프라인 엔드투엔드 점검
- 예비 리스크 대응 시간 (여기까지 밀릴 걸 미리 감안)

## Phase 9 — 문서화 및 마무리 (1일)

- README: `Base Model → 내 데이터셋 → Fine-tuning → 평가 → 경량화/배포` 흐름 명시
- 아키텍처 다이어그램
- 데모 스크린샷/GIF, 파인튜닝 전후 비교 결과 정리

---

## 총 소요 예상: 약 11~13일 (1~2주 범위 내)

시간이 부족해지면 잘라내는 순서(뒤로 갈수록 먼저 잘라도 됨):
1. NestJS/Spring 레이어 → FastAPI가 직접 처리하도록 단순화
2. pgvector 유사 뉴스 검색 (확장 기능)
3. Open WebUI 데모 (README 설명으로 대체 가능)
4. Qwen3 파인튜닝 데이터셋 규모를 더 줄임 (완성 자체는 유지)

절대 자르면 안 되는 것: Phase 1~3(핵심 데이터-분류 파이프라인), Phase 9(README 스토리텔링)
