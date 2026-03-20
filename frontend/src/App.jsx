import { useState } from "react"
import ResultCard from "./components/ResultCard"
import Charts from "./components/Charts"

export default function App() {
  const [url, setUrl]       = useState("")
  const [results, setResults] = useState([])
  const [stats, setStats]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function handleAnalyze() {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setResults([])
    setStats(null)

    try {
      const res = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Something went wrong")
      }
      const data = await res.json()
      setResults(data.results)
      setStats(data.stats)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <div className="header-accent" />
        <h1>Aspect-Based Sentiment Analysis</h1>
        <p className="subtitle">Analyze Nepali YouTube comments in real time</p>
      </div>

      <div className="input-section">
        <div className="input-row">
          <input
            type="text"
            placeholder="Paste a YouTube URL..."
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAnalyze()}
            disabled={loading}
          />
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Fetching and analyzing comments — this takes about 60–90 seconds</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="results">
          <p className="result-count">{results.length} comments analyzed</p>
          {results.map((r, i) => <ResultCard key={i} index={i} result={r} />)}
        </div>
      )}

      <Charts stats={stats} />
    </div>
  )
}