# SentiQuant

한국 주식 뉴스의 논조(긍정/부정/중립)를 분류하고, 그 논조가 실제 주가와 관련이 있는지 **정직하게** 검증한 개인 프로젝트입니다.

> "뉴스로 내일 주가를 맞힐 수 있는가"라는 첫 가설은 기각됐습니다. 대신 "이 기사의 논조가 무엇인가"로 과제를 다시 정의했고, 그 결과를 과대포장하지 않기 위해 스스로 발견한 평가 버그를 고치고, 통계적으로 유의하지만 작은 효과크기를 있는 그대로 보고했습니다.

전체 스토리(그래프·스크린샷 포함)는 [`portfolio/sentiquant.html`](portfolio/sentiquant.html)에, 날짜별 상세 작업 기록은 [WORKLOG.md](WORKLOG.md)에 있습니다.

## 핵심 결과

| 항목 | 값 |
|---|---|
| 논조 분류 정확도 (test) | **83.5%** (F1 macro 82.2%) — 주가 직접 예측 방식(최고 38.5%) 대비 2배 이상 |
| 수집 종목 | 20개 (7개 섹터 분산) |
| 수집 뉴스 | 34,000+ 건 (NAVER 뉴스검색 API) |
| 최종 학습 표본 | 24,695건 (Qwen3-4B 자동 논조 라벨링) |
| 논조-주가 상관관계 | 통계적으로 유의(p<0.0001)하나 **효과크기는 작음**(r²<2%) — 매매 신호로 쓸 근거는 아님 |

## 이 프로젝트가 보여주는 것

- **실패한 가설을 인정하고 방향을 바꾼 판단력**: 뉴스 헤드라인으로 익일 주가 방향을 맞히려던 첫 시도는 찍기 확률 수준에서 벗어나지 못했습니다. 원인을 분석해 "라벨을 주가가 아니라 텍스트 자체의 논조로 재정의"하는 방향으로 전환했고, 정확도가 2배 이상 뛰었습니다.
- **자기 검증 능력**: 종목별 성능을 점검하다가 기존 평가 방법론(train/val/test 분할)이 특정 종목들을 검증셋에서 통째로 누락시키는 버그를 스스로 발견하고 수정했습니다. "81.3%"라는 초기 결과가 사실은 일부 종목에만 편중된 수치였다는 것을 밝히고 바로잡았습니다.
- **정직한 통계 보고**: 논조와 실제 수익률의 상관관계를 여러 시차(t+1~t+10)로 나눠 검증하며, 처음엔 과장되게 해석했던 결과("역인과관계가 확인됐다")를 효과크기 기준(Cohen's r)으로 다시 검토해 "약한 신호" 수준으로 스스로 정정했습니다.
- **실전 도구화(RAG)**: 분류기 하나로 끝내지 않고, 임베딩 기반 유사사례 검색 + Qwen3-4B 설명 생성을 결합한 RAG 파이프라인을 FastAPI + Next.js로 구현했습니다. LLM 프롬프트에도 "확정적 예측처럼 말하지 말 것"을 명시해, 도구 자체가 결과를 과장하지 않도록 설계했습니다.

## 시스템 구성

```
NAVER 뉴스 API ─┐
                ├─▶ 뉴스-주가 매칭 ─▶ Qwen3-4B 논조 라벨링 ─▶ KLUE-BERT 파인튜닝(분류기)
yfinance(주가) ─┘                                                    │
                                                                      ▼
                                        임베딩 검색(과거 유사사례 + 실제 수익률)
                                                                      │
                                                                      ▼
                                    FastAPI(/analyze, /explain) ─▶ Next.js 웹 UI
```

## 기술 스택

**데이터 · ML** Python · pandas · scikit-learn · Transformers · Unsloth · Qwen3-4B · KLUE-BERT · sentence-transformers
**백엔드 · 프론트** FastAPI · Next.js · React · TypeScript
**기타** NAVER API HUB · yfinance · RTX 4060 Ti(로컬 4bit 추론)

## 더 읽어보기

- [portfolio/sentiquant.html](portfolio/sentiquant.html) — 그래프·스크린샷과 함께 정리한 전체 스토리 (가장 먼저 보기 좋음)
- [WORKLOG.md](WORKLOG.md) — 날짜별 상세 작업 기록, 시행착오, 의사결정 과정
- [DATA_PIPELINE_RULES.md](DATA_PIPELINE_RULES.md) — 데이터 파이프라인 재실행 순서/체크리스트
- [RAG_DEMO.md](RAG_DEMO.md) — RAG 데모 아키텍처 상세
- [project_roadmap.md](project_roadmap.md) — 원래 계획했던 전체 로드맵

## 로컬 실행

```bash
# 1. 환경
conda activate unsloth_env   # Python 3.12, PyTorch 2.11 + CUDA 13.0
pip install -r requirements.txt   # (또는 DATA_PIPELINE_RULES.md 참고해 개별 설치)

# 2. .env에 NAVER API 키 필요 (NAVER_API_KEY_ID, NAVER_API_KEY)

# 3. RAG 데모 실행 (사전 학습된 데이터/모델 필요 — DATA_PIPELINE_RULES.md 참고)
cd api && uvicorn main:app --reload        # :8000
cd web && npm install && npm run dev        # :3000
```

`data/`, `models/`는 용량 문제로 git에 포함하지 않았습니다. 재현하려면 DATA_PIPELINE_RULES.md의 순서대로 파이프라인을 처음부터 돌려야 합니다.

## 연락채널및 기타 작업정보

[github.com/dgmans21](https://github.com/dgmans21) · dgdtkjk21@gmail.com
