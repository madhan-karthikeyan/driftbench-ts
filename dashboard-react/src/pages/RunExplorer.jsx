import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Bar, Scatter, ReferenceLine, Area, AreaChart
} from 'recharts'
import { ArrowLeft, AlertTriangle, TrendingDown, RotateCcw, Activity, Eye, CheckCircle, XCircle } from 'lucide-react'
import { MODEL_COLORS, STRATEGY_COLORS, MODEL_LABELS, STRATEGY_LABELS } from '../types/constants'
import api from '../services/api'

export default function RunExplorer() {
  const { dataset, strategy, model } = useParams()
  const [run, setRun] = useState(null)
  const [windows, setWindows] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.getRun(dataset, strategy, model),
      api.getRunWindows(dataset, strategy, model)
    ]).then(([runData, windowData]) => {
      setRun(runData)
      setWindows(windowData.windows || [])
    }).catch(console.error)
      .finally(() => setLoading(false))
  }, [dataset, strategy, model])

  if (loading) return <LoadingSkeleton />

  if (!run || run.error) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-500">Run not found</p>
        <Link to="/" className="text-blue-400 hover:underline mt-2 inline-block">Back to Overview</Link>
      </div>
    )
  }

  const { metrics } = run

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to={`/dataset/${dataset}`} className="p-2 rounded-lg bg-dark-700 hover:bg-dark-600 transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <span className="capitalize">{dataset}</span>
            <span className="text-slate-500">/</span>
            <span 
              className="badge text-white"
              style={{ backgroundColor: STRATEGY_COLORS[strategy] }}
            >
              {STRATEGY_LABELS[strategy]}
            </span>
            <span 
              className="badge"
              style={{ backgroundColor: MODEL_COLORS[model], color: model === 'seasonal_naive' ? 'black' : 'white' }}
            >
              {MODEL_LABELS[model]}
            </span>
          </h1>
          <p className="text-slate-400 mt-1">Detailed window-level analysis</p>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
          icon={<TrendingDown className="w-5 h-5" />}
          label="MAE"
          value={metrics.mae?.toFixed(2)}
          color="text-blue-400"
        />
        <MetricCard 
          icon={<Activity className="w-5 h-5" />}
          label="RMSE"
          value={metrics.rmse?.toFixed(2)}
          color="text-purple-400"
        />
        <MetricCard 
          icon={<RotateCcw className="w-5 h-5" />}
          label="Retrains"
          value={metrics.retrain_total || 0}
          color="text-amber-400"
        />
        <MetricCard 
          icon={<AlertTriangle className="w-5 h-5" />}
          label="Drift Rate"
          value={`${metrics.drift_detection_rate?.toFixed(2) || 0}%`}
          color="text-red-400"
        />
      </div>

      {/* Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InsightCard
          icon={<CheckCircle className="w-5 h-5 text-green-400" />}
          label="Drift Detection Accuracy"
          value={calculateDriftAccuracy(windows)}
          subtext="Correctly identified vs missed"
        />
        <InsightCard
          icon={<XCircle className="w-5 h-5 text-red-400" />}
          label="False Positive Rate"
          value={calculateFalsePositiveRate(windows)}
          subtext="Unnecessary detections"
        />
        <InsightInsightCard
          icon={<Activity className="w-5 h-5 text-blue-400" />}
          label="Recovery Time"
          value={calculateRecoveryTime(windows)}
          subtext="Windows to stabilize after drift"
        />
      </div>

      {/* Error Timeline with Drift Markers */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-blue-400" />
            Error Timeline with Drift & Retrain Events
          </h2>
        </div>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={windows} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <defs>
                <linearGradient id="errorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="step" 
                tick={{ fill: '#94a3b8' }}
                label={{ value: 'Time Window', position: 'bottom', fill: '#94a3b8' }}
              />
              <YAxis yAxisId="left" tick={{ fill: '#94a3b8' }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tick={{ fill: '#94a3b8' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
              <Legend />
              <Area 
                yAxisId="left"
                type="monotone"
                dataKey="mean_error" 
                name="Mean Error"
                fill="url(#errorGradient)"
                stroke="#3b82f6"
                strokeWidth={2}
              />
              <Line 
                yAxisId="left"
                type="monotone"
                dataKey="max_error" 
                name="Max Error"
                stroke="#ef4444"
                strokeWidth={1}
                dot={false}
              />
              <Scatter
                yAxisId="right"
                dataKey="drift_detected"
                name="Drift Detected"
                fill="#eab308"
                shape={(props) => {
                  if (!props.payload.drift_detected) return null
                  return (
                    <polygon 
                      points={`${props.cx},${props.cy - 8} ${props.cx + 7},${props.cy + 4} ${props.cx - 7},${props.cy + 4}`}
                      fill="#eab308"
                    />
                  )
                }}
              />
              <Scatter
                yAxisId="right"
                dataKey="retrained"
                name="Retrained"
                fill="#22c55e"
                shape={(props) => {
                  if (!props.payload.retrained) return null
                  return (
                    <rect 
                      x={props.cx - 6}
                      y={props.cy - 6}
                      width={12}
                      height={12}
                      fill="#22c55e"
                      rx={2}
                    />
                  )
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-400 rounded" />
            <span className="text-slate-400">Error (area = avg, line = max)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-0 h-0 border-l-[6px] border-r-[6px] border-b-[10px] border-l-transparent border-r-transparent border-b-amber-400" />
            <span className="text-slate-400">Drift Detected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-sm" />
            <span className="text-slate-400">Retrain Event</span>
          </div>
        </div>
      </div>

      {/* Drift Score Timeline */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Drift Score Over Time
          </h2>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={windows} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <defs>
                <linearGradient id="driftGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#eab308" stopOpacity={0.5}/>
                  <stop offset="95%" stopColor="#eab308" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" tick={{ fill: '#94a3b8' }} />
              <YAxis tick={{ fill: '#94a3b8' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="5,5" label="Drift Threshold" />
              <Area 
                type="monotone"
                dataKey="drift_score" 
                name="Drift Score"
                fill="url(#driftGradient)"
                stroke="#eab308"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Window Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center gap-2">
            <Eye className="w-5 h-5 text-slate-400" />
            Window-Level Metrics
          </h2>
        </div>
        <div className="table-container max-h-96">
          <table className="w-full">
            <thead className="table-header sticky top-0">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400">Step</th>
                <th className="px-3 py-2 text-right text-xs font-semibold text-slate-400">Mean Error</th>
                <th className="px-3 py-2 text-right text-xs font-semibold text-slate-400">Max Error</th>
                <th className="px-3 py-2 text-right text-xs font-semibold text-slate-400">Std</th>
                <th className="px-3 py-2 text-center text-xs font-semibold text-slate-400">Drift</th>
                <th className="px-3 py-2 text-center text-xs font-semibold text-slate-400">Retrain</th>
                <th className="px-3 py-2 text-right text-xs font-semibold text-slate-400">Drift Score</th>
              </tr>
            </thead>
            <tbody>
              {windows.slice(0, 100).map((w, i) => (
                <tr key={i} className="table-row">
                  <td className="px-3 py-2 font-mono text-slate-400">{w.step}</td>
                  <td className="px-3 py-2 text-right font-mono text-blue-400">{w.mean_error?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono text-red-400">{w.max_error?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono text-slate-400">{w.std_error?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-center">
                    {w.drift_detected ? (
                      <span className="inline-flex items-center justify-center w-5 h-5 bg-amber-500 rounded text-black">
                        <AlertTriangle className="w-3 h-3" />
                      </span>
                    ) : (
                      <span className="text-slate-600">-</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-center">
                    {w.retrained ? (
                      <span className="inline-flex items-center justify-center w-5 h-5 bg-green-500 rounded text-white">
                        <RotateCcw className="w-3 h-3" />
                      </span>
                    ) : (
                      <span className="text-slate-600">-</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className={`font-mono ${
                      w.drift_score > 0.5 ? 'text-red-400' : w.drift_score > 0.3 ? 'text-amber-400' : 'text-slate-400'
                    }`}>
                      {w.drift_score?.toFixed(3)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {windows.length > 100 && (
          <div className="text-center py-2 text-sm text-slate-500 border-t border-dark-700">
            Showing 100 of {windows.length} windows
          </div>
        )}
      </div>
    </div>
  )
}

// Helper components
function MetricCard({ icon, label, value, color }) {
  return (
    <div className="card">
      <div className={`${color} mb-2`}>{icon}</div>
      <div className="text-3xl font-bold text-white">{value}</div>
      <div className="text-sm text-slate-400">{label}</div>
    </div>
  )
}

function InsightCard({ icon, label, value, subtext }) {
  return (
    <div className="card">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <div className="text-2xl font-bold text-white">{value}</div>
          <div className="text-sm text-slate-400">{label}</div>
          {subtext && <div className="text-xs text-slate-500">{subtext}</div>}
        </div>
      </div>
    </div>
  )
}

function InsightInsightCard({ icon, label, value, subtext }) {
  return (
    <div className="card">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <div className="text-2xl font-bold text-white">{value}</div>
          <div className="text-sm text-slate-400">{label}</div>
          {subtext && <div className="text-xs text-slate-500">{subtext}</div>}
        </div>
      </div>
    </div>
  )
}

// Analysis functions
function calculateDriftAccuracy(windows) {
  const detected = windows.filter(w => w.drift_detected).length
  const total = windows.length
  return `${detected} events (${((detected / total) * 100).toFixed(1)}%)`
}

function calculateFalsePositiveRate(windows) {
  const detected = windows.filter(w => w.drift_detected).length
  const highScore = windows.filter(w => w.drift_score > 0.7).length
  const fp = Math.max(0, detected - highScore)
  return `${fp} (${((fp / Math.max(detected, 1)) * 100).toFixed(1)}%)`
}

function calculateRecoveryTime(windows) {
  let recoveryTime = 0
  let inDrift = false
  for (const w of windows) {
    if (w.drift_detected) {
      inDrift = true
      recoveryTime = 0
    } else if (inDrift) {
      recoveryTime++
      if (recoveryTime >= 5) {
        inDrift = false
        return `${recoveryTime} windows`
      }
    }
  }
  return recoveryTime > 0 ? `${recoveryTime} windows` : 'N/A'
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-96 bg-dark-700 rounded animate-pulse" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-24 bg-dark-700 rounded-xl animate-pulse" />
        ))}
      </div>
      <div className="h-96 bg-dark-700 rounded-xl animate-pulse" />
    </div>
  )
}
