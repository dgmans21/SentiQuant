"use client";

import { useState } from "react";

type SimilarCase = {
  similarity: number;
  stock: string;
  title: string;
  qwen_label: string;
  excess_t1: number | null;
  excess_t3: number | null;
  excess_t5: number | null;
  excess_t10: number | null;
};

type AnalyzeResponse = {
  label: string;
  confidence: number;
  similar_cases: SimilarCase[];
  historical_avg: Record<string, number | null>;
};

const API_URL = "http://127.0.0.1:8000";

const EXAMPLES = [
  {
    label: "긍정 예시",
    text: "삼성전자, 3분기 영업이익 시장 예상치 상회하며 실적 개선세",
  },
  {
    label: "부정 예시",
    text: "SK하이닉스, 반도체 업황 둔화 우려에 주가 하락",
  },
  {
    label: "자사주 매입 (맥락 이해 확인용)",
    text: "당사는 자기주식 500억원 규모를 신규 취득하기로 결정하였습니다",
  },
  {
    label: "절차성 공시 (중립 기대)",
    text: "당사는 이사회 결의를 통해 임시주주총회 소집을 결정하였습니다",
  },
  {
    label: "장기 데이터 예시 (그래프 4칸 다 채워짐)",
    text: "신한지주, 이자이익 감소와 대손충당금 증가로 순이익 둔화",
  },
];

function HorizonBarChart({ avg }: { avg: Record<string, number | null> }) {
  const horizons = [
    { key: "excess_t1", label: "다음날" },
    { key: "excess_t3", label: "3일 후" },
    { key: "excess_t5", label: "5일 후" },
    { key: "excess_t10", label: "10일 후" },
  ];
  const scale = Math.max(1, ...horizons.map((h) => Math.abs(avg[h.key] ?? 0)));

  return (
    <div style={{ marginTop: 8 }}>
      {horizons.map((h) => {
        const v = avg[h.key];
        const pct = v === null || v === undefined ? 0 : (Math.abs(v) / scale) * 50;
        const isNeg = (v ?? 0) < 0;
        return (
          <div key={h.key} style={{ display: "flex", alignItems: "center", height: 24, fontSize: 12 }}>
            <div style={{ width: 52 }}>{h.label}</div>
            <div style={{ flex: 1, position: "relative", height: 16, background: "#f0f0f0" }}>
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#999" }} />
              {v !== null && v !== undefined && (
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    bottom: 0,
                    left: isNeg ? `${50 - pct}%` : "50%",
                    width: `${pct}%`,
                    background: isNeg ? "#e05353" : "#3aa76d",
                  }}
                />
              )}
            </div>
            <div style={{ width: 60, textAlign: "right" }}>
              {v === null || v === undefined ? "데이터 없음" : `${v.toFixed(2)}%`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);
    setExplainError(null);
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`서버 오류 (${res.status})`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "알 수 없는 오류");
    } finally {
      setLoading(false);
    }
  };

  const fmtPct = (v: number | null) => (v === null || v === undefined ? "-" : `${v.toFixed(2)}%`);

  const requestExplanation = async () => {
    if (!text.trim()) return;
    setExplaining(true);
    setExplainError(null);
    try {
      const res = await fetch(`${API_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`서버 오류 (${res.status})`);
      const data = await res.json();
      setExplanation(data.explanation);
    } catch (e) {
      setExplainError(e instanceof Error ? e.message : "알 수 없는 오류");
    } finally {
      setExplaining(false);
    }
  };

  return (
    <main>
      <h1>SentiQuant 논조 분석 데모</h1>
      <p style={{ color: "#666", fontSize: 14 }}>
        뉴스 기사 제목/본문을 붙여넣으면 논조를 분류하고, 과거 유사 기사의 실제 수익률을 참고로 보여줍니다.
        <br />
        <b>주의: 예측이 아니라 과거 통계 참고용입니다.</b>
      </p>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => setText(ex.text)}
            style={{ padding: "4px 10px", fontSize: 12, background: "#eee", border: "1px solid #ccc", borderRadius: 4 }}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        style={{ width: "100%", padding: 8, fontSize: 14 }}
        placeholder="예: 삼성전자, 3분기 영업이익 시장 예상치 상회하며 실적 개선세"
      />
      <button onClick={analyze} disabled={loading} style={{ marginTop: 8, padding: "8px 16px" }}>
        {loading ? "분석 중..." : "분석하기"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 24 }}>
          <h2>
            논조:{" "}
            <span
              style={{
                color: result.label === "긍정" ? "green" : result.label === "부정" ? "crimson" : "gray",
              }}
            >
              {result.label}
            </span>{" "}
            (확신도 {(result.confidence * 100).toFixed(1)}%)
          </h2>

          <h3>참고: 유사 과거 사례 평균 초과수익률 (예측 아닌 통계)</h3>
          <p style={{ fontSize: 11, color: "#999", margin: "0 0 4px" }}>
            기사 발행일 기준, 주식시장이 열린 날(거래일)로 센 날짜입니다. 코스피 지수 대비 초과 수익률.
          </p>
          <HorizonBarChart avg={result.historical_avg} />

          <h3>유사 과거 기사</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>
                <th>유사도</th>
                <th>종목</th>
                <th>제목</th>
                <th>논조</th>
                <th>다음날</th>
                <th>3일 후</th>
              </tr>
            </thead>
            <tbody>
              {result.similar_cases.map((c, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #eee" }}>
                  <td>{c.similarity.toFixed(2)}</td>
                  <td>{c.stock}</td>
                  <td>{c.title}</td>
                  <td>{c.qwen_label}</td>
                  <td>{fmtPct(c.excess_t1)}</td>
                  <td>{fmtPct(c.excess_t3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 16 }}>
            <button onClick={requestExplanation} disabled={explaining} style={{ padding: "8px 16px" }}>
              {explaining ? "Qwen이 분석 중... (수초~수십초 소요)" : "추가 분석 (Qwen)"}
            </button>
            {explainError && <p style={{ color: "red" }}>{explainError}</p>}
            {explanation && (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  background: "#f7f7f7",
                  borderRadius: 6,
                  whiteSpace: "pre-wrap",
                  fontSize: 14,
                  lineHeight: 1.6,
                }}
              >
                {explanation}
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
