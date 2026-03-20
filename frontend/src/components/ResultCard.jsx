export default function ResultCard({ result, index }) {
  const sentimentConfig = {
    positive: { bg: "#FFF7ED", color: "#C2410C", label: "Positive" },
    negative: { bg: "#FFF1F2", color: "#BE123C", label: "Negative" },
    neutral:  { bg: "#F5F5F4", color: "#57534E", label: "Neutral"  },
  }

  if (result.skipped) {
    return (
      <div className="card">
        <div className="card-index">#{index + 1}</div>
        <div className="card-body">
          <p className="comment-text" style={{ color: "#aaa" }}>
            {result.original_comment}
          </p>
          <div className="card-footer">
            <div className="badge" style={{ background: "#F5F5F4", color: "#999" }}>
              {result.language} — skipped
            </div>
          </div>
        </div>
      </div>
    )
  }

  const s = sentimentConfig[result.sentiment] || sentimentConfig.neutral

  return (
    <div className="card">
      <div className="card-index">#{index + 1}</div>
      <div className="card-body">
        <div className="comment-row">
          <div className="comment-block">
            <span className="label">Original ({result.language})</span>
            <p className="comment-text">{result.original_comment}</p>
          </div>
          <div className="arrow">→</div>
          <div className="comment-block">
            <span className="label">Devanagari</span>
            <p className="comment-text devanagari">{result.devanagari_comment}</p>
          </div>
        </div>
        <div className="card-footer">
          <div className="badge aspect">
            <span className="badge-dot aspect-dot" />
            {result.aspect}
          </div>
          <div className="badge sentiment" style={{ background: s.bg, color: s.color }}>
            <span className="badge-dot" style={{ background: s.color }} />
            {s.label}
          </div>
        </div>
      </div>
    </div>
  )
}
