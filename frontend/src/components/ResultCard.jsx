export default function ResultCard({ result, index }) {
  const sentimentConfig = {
    positive: { bg: "#F0FDF4", color: "#15803D", label: "Positive" },
    negative: { bg: "#FEF2F2", color: "#B91C1C", label: "Negative" },
    neutral:  { bg: "#F5F5F4", color: "#57534E", label: "Neutral"  },
  }

  const languageLabel = {
    "NE_DEV":     "Nepali Devanagari",
    "NE_ROM":     "Roman Nepali",
    "EN":         "English",
    "CODE_MIXED": "Code Mixed",
    "UNKNOWN":    "Unknown",
  }

  const getLanguageLabel = (lang) => languageLabel[lang] || lang

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
              {getLanguageLabel(result.language)} — skipped
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-index">#{index + 1}</div>
      <div className="card-body">

        <div className="comment-row">
          <div className="comment-block">
            <span className="label">Original ({getLanguageLabel(result.language)})</span>
            <p className="comment-text">{result.original_comment}</p>
          </div>
          <div className="arrow">→</div>
          <div className="comment-block">
            <span className="label">Devanagari</span>
            <p className="comment-text devanagari">{result.devanagari_comment}</p>
          </div>
        </div>

        <div className="targets-section">
          {result.targets && result.targets.length > 0 ? (
            result.targets.map((t, i) => {
              const s = sentimentConfig[t.sentiment] || sentimentConfig.neutral
              return (
                <div key={i} className="target-row">
                  <div className="badge target-badge">
                    {t.target}
                  </div>
                  <div className="badge aspect">
                    <span className="badge-dot aspect-dot" />
                    {t.aspect}
                  </div>
                  <div className="badge sentiment"
                    style={{ background: s.bg, color: s.color }}>
                    <span className="badge-dot"
                      style={{ background: s.color }} />
                    {s.label}
                  </div>
                </div>
              )
            })
          ) : (
            <p style={{ fontSize: 13, color: "#aaa" }}>No targets identified</p>
          )}
        </div>

      </div>
    </div>
  )
}