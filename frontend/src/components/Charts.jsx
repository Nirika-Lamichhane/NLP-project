import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts"

const SENTIMENT_COLORS = {
  Positive: "#F97316",
  Negative: "#1D4ED8",
  Neutral:  "#A8A29E",
}

const SAMPLE_SENTIMENT = [
  { name: "Positive", value: 5, key: "positive" },
  { name: "Negative", value: 3, key: "negative" },
  { name: "Neutral",  value: 2, key: "neutral"  },
]

const SAMPLE_ASPECT = [
  { name: "Service",    value: 4 },
  { name: "Policy",     value: 3 },
  { name: "Governance", value: 2 },
  { name: "Corruption", value: 2 },
  { name: "Economy",    value: 1 },
]

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: "white",
        border: "1px solid #F0E8DF",
        borderRadius: 8,
        padding: "8px 14px",
        fontSize: 13,
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
      }}>
        <p style={{ fontWeight: 600, marginBottom: 2 }}>{payload[0].name}</p>
        <p style={{ color: "#F97316" }}>{payload[0].value} comments</p>
      </div>
    )
  }
  return null
}

export default function Charts({ results }) {
  const isSample = !results || results.length === 0

  const sentimentData = isSample
    ? SAMPLE_SENTIMENT
    : Object.entries(
        results.reduce((acc, r) => {
          const key = r.sentiment.charAt(0).toUpperCase() + r.sentiment.slice(1)
          acc[key] = (acc[key] || 0) + 1
          return acc
        }, {})
      ).map(([name, value]) => ({ name, value, key: name.toLowerCase() }))

  const aspectData = isSample
    ? SAMPLE_ASPECT
    : Object.entries(
        results.reduce((acc, r) => {
          const key = r.aspect.charAt(0).toUpperCase() + r.aspect.slice(1)
          acc[key] = (acc[key] || 0) + 1
          return acc
        }, {})
      )
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)

  return (
    <div className="charts-section">
      <div className="charts-header">
        <h2 className="charts-title">Analysis Summary</h2>
        {isSample && (
          <span className="sample-badge">Sample data — run analysis to see real results</span>
        )}
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3 className="chart-label">Sentiment Breakdown</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%"
                cy="50%"
                innerRadius={65}
                outerRadius={105}
                paddingAngle={3}
                dataKey="value"
              >
                {sentimentData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={SENTIMENT_COLORS[entry.name] || "#ccc"}
                    opacity={isSample ? 0.4 : 1}
                  />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend
                formatter={(value) => (
                  <span style={{ fontSize: 13, color: "#555" }}>{value}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3 className="chart-label">Aspect Breakdown</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={aspectData}
              margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
            >
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: "#888" }}
                interval={0}
                angle={-25}
                textAnchor="end"
                height={50}
              />
              <YAxis tick={{ fontSize: 12, fill: "#888" }} allowDecimals={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                dataKey="value"
                fill="#F97316"
                radius={[5, 5, 0, 0]}
                opacity={isSample ? 0.4 : 1}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}