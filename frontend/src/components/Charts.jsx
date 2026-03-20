import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Legend
} from "recharts"

const SENTIMENT_COLORS = {
  Positive: "#F97316",
  Negative: "#1D4ED8",
  Neutral:  "#A8A29E",
}

const SAMPLE_SENTIMENT = [
  { name: "Positive", value: 5 },
  { name: "Negative", value: 3 },
  { name: "Neutral",  value: 2 },
]

const SAMPLE_ASPECT = [
  { name: "Service",    value: 4 },
  { name: "Policy",     value: 3 },
  { name: "Governance", value: 2 },
  { name: "Corruption", value: 2 },
  { name: "Economy",    value: 1 },
]

const SAMPLE_TARGETS = [
  {
    target: "Balen",
    mentions: 8,
    breakdown: [
      { aspect: "Governance", positive: 4, negative: 1, neutral: 1 },
      { aspect: "Economy",    positive: 3, negative: 1, neutral: 0 },
      { aspect: "Service",    positive: 2, negative: 0, neutral: 1 },
    ]
  },
  {
    target: "RSP",
    mentions: 5,
    breakdown: [
      { aspect: "Policy",     positive: 1, negative: 3, neutral: 0 },
      { aspect: "Corruption", positive: 0, negative: 2, neutral: 1 },
      { aspect: "Governance", positive: 1, negative: 1, neutral: 0 },
    ]
  },
]

const TARGET_PILLS = ["#FFF7ED", "#EFF6FF"]
const TARGET_TEXT  = ["#C2410C", "#1D4ED8"]

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: "white",
        border: "1px solid #F0E8DF",
        borderRadius: 8,
        padding: "8px 14px",
        fontSize: 13,
      }}>
        <p style={{ fontWeight: 600, marginBottom: 4 }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.fill }}>
            {p.name}: {p.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

function OverallCharts({ sentimentData, aspectData, isSample }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
        <h2 className="charts-title" style={{ marginBottom: 0 }}>Overall summary</h2>
        {isSample && <span className="sample-badge">Sample data</span>}
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3 className="chart-label">Sentiment breakdown</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%" cy="50%"
                innerRadius={65} outerRadius={105}
                paddingAngle={3} dataKey="value"
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
              <Legend formatter={v => <span style={{ fontSize: 13, color: "#555" }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3 className="chart-label">Aspect breakdown</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={aspectData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: "#888" }}
                interval={0} angle={-25}
                textAnchor="end" height={50}
              />
              <YAxis tick={{ fontSize: 12, fill: "#888" }} allowDecimals={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                dataKey="value" fill="#F97316"
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

function TargetChart({ targetData, index, isSample }) {
  const pillBg   = TARGET_PILLS[index] || "#F5F5F4"
  const pillText = TARGET_TEXT[index]  || "#555"

  return (
    <div className="chart-card">
      <div style={{
        display: "inline-block",
        fontSize: 12, fontWeight: 600,
        padding: "3px 10px", borderRadius: 20,
        background: pillBg, color: pillText,
        marginBottom: 10,
        opacity: isSample ? 0.5 : 1,
      }}>
        {targetData.target} — {targetData.mentions} mentions
      </div>
      <h3 className="chart-label">Top 3 aspects</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={targetData.breakdown}
          margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
        >
          <XAxis
            dataKey="aspect"
            tick={{ fontSize: 11, fill: "#888" }}
            interval={0} angle={-15}
            textAnchor="end" height={45}
          />
          <YAxis tick={{ fontSize: 11, fill: "#888" }} allowDecimals={false} />
          <Tooltip content={<CustomTooltip />} />
          <Legend formatter={v => <span style={{ fontSize: 12, color: "#555" }}>{v}</span>} />
          <Bar dataKey="positive" name="Positive" fill="#22C55E" radius={[3, 3, 0, 0]} opacity={isSample ? 0.4 : 1} />
          <Bar dataKey="negative" name="Negative" fill="#EF4444" radius={[3, 3, 0, 0]} opacity={isSample ? 0.4 : 1} />
          <Bar dataKey="neutral"  name="Neutral"  fill="#9CA3AF" radius={[3, 3, 0, 0]} opacity={isSample ? 0.4 : 1} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Charts({ stats }) {
  const isSample = !stats

  const sentimentData = isSample
    ? SAMPLE_SENTIMENT
    : Object.entries(stats.sentiment_counts).map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
      }))

  const aspectData = isSample
    ? SAMPLE_ASPECT
    : Object.entries(stats.aspect_counts)
        .map(([name, value]) => ({
          name: name.charAt(0).toUpperCase() + name.slice(1),
          value,
        }))
        .sort((a, b) => b.value - a.value)

  const targetData = isSample
    ? SAMPLE_TARGETS
    : stats.per_target.map(t => ({
        ...t,
        breakdown: t.breakdown.map(b => ({
          ...b,
          aspect: b.aspect.charAt(0).toUpperCase() + b.aspect.slice(1),
        }))
      }))

  return (
    <div className="charts-section">

      <OverallCharts
        sentimentData={sentimentData}
        aspectData={aspectData}
        isSample={isSample}
      />

      <div style={{ marginTop: 40 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
          <h2 className="charts-title" style={{ marginBottom: 0 }}>Per-target breakdown</h2>
          {isSample && <span className="sample-badge">Sample data</span>}
        </div>
        <div className="charts-grid">
          {targetData.map((t, i) => (
            <TargetChart key={i} targetData={t} index={i} isSample={isSample} />
          ))}
        </div>
      </div>

    </div>
  )
}