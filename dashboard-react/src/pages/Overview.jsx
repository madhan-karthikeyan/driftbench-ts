import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { BarChart3, Database, Cpu, Trophy, ArrowRight, TrendingDown, RotateCcw } from 'lucide-react'
import { MODEL_COLORS, STRATEGY_COLORS, MODEL_LABELS, STRATEGY_LABELS } from '../types/constants'
import api from '../services/api'

export default function Overview() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({ dataset: 'all', model: 'all', strategy: 'all' })

  useEffect(() => {
    api.getOverview()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <LoadingSkeleton />
  }

  if (!data) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-500">No results found. Run experiments first.</p>
      </div>
    )
  }

  const { summary, runs, datasets, models, strategies } = data

  const filteredRuns = runs.filter(run => {
    if (filters.dataset !== 'all' && run.dataset !== filters.dataset) return false
    if (filters.model !== 'all' && run.model !== filters.model) return false
    if (filters.strategy !== 'all' && run.strategy !== filters.strategy) return false
    return true
  })

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Experiment Overview</h1>
          <p className="text-slate-400 mt-1">Dataset × Strategy × Model benchmarking results</p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard 
          icon={<Database className="w-6 h-6" />}
          label="Datasets"
          value={summary.total_datasets}
          subtext={datasets.join(', ')}
          color="text-blue-400"
        />
        <SummaryCard 
          icon={<Cpu className="w-6 h-6" />}
          label="Models"
          value={summary.total_models}
          subtext={models.length + ' variants tested'}
          color="text-purple-400"
        />
        <SummaryCard 
          icon={<TrendingDown className="w-6 h-6" />}
          label="Best Model"
          value={MODEL_LABELS[summary.best_model] || summary.best_model}
          subtext={`MAE: ${summary.best_model_mae}`}
          color="text-green-400"
        />
        <SummaryCard 
          icon={<RotateCcw className="w-6 h-6" />}
          label="Total Runs"
          value={summary.total_runs}
          subtext="experiments completed"
          color="text-amber-400"
        />
      </div>

      {/* Best Per Dataset */}
      {data.best_per_dataset && data.best_per_dataset.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-400" />
              Best Strategy per Dataset
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {data.best_per_dataset.map(item => (
              <div key={item.dataset} className="bg-dark-700 rounded-lg p-4">
                <div className="text-lg font-semibold text-white capitalize mb-2">{item.dataset}</div>
                <div className="flex items-center gap-2 mb-1">
                  <span 
                    className="badge"
                    style={{ backgroundColor: STRATEGY_COLORS[item.best_strategy] }}
                  >
                    {STRATEGY_LABELS[item.best_strategy]}
                  </span>
                  <span 
                    className="badge"
                    style={{ backgroundColor: MODEL_COLORS[item.best_model] }}
                  >
                    {MODEL_LABELS[item.best_model]}
                  </span>
                </div>
                <div className="text-2xl font-bold text-green-400">{item.best_mae.toFixed(1)}</div>
                <div className="text-xs text-slate-500">MAE • {item.improvement.toFixed(1)}% better than worst</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap gap-4 items-center">
          <span className="text-slate-400 text-sm">Filter:</span>
          <select 
            className="select"
            value={filters.dataset}
            onChange={e => setFilters(f => ({ ...f, dataset: e.target.value }))}
          >
            <option value="all">All Datasets</option>
            {datasets.map(ds => <option key={ds} value={ds}>{ds}</option>)}
          </select>
          <select 
            className="select"
            value={filters.model}
            onChange={e => setFilters(f => ({ ...f, model: e.target.value }))}
          >
            <option value="all">All Models</option>
            {models.map(m => <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>)}
          </select>
          <select 
            className="select"
            value={filters.strategy}
            onChange={e => setFilters(f => ({ ...f, strategy: e.target.value }))}
          >
            <option value="all">All Strategies</option>
            {strategies.map(s => <option key={s} value={s}>{STRATEGY_LABELS[s] || s}</option>)}
          </select>
          <span className="text-slate-500 text-sm ml-auto">{filteredRuns.length} results</span>
        </div>
      </div>

      {/* Results Table */}
      <div className="card">
        <div className="table-container">
          <table className="w-full">
            <thead className="table-header">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase">Dataset</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase">Strategy</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase">Model</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase">MAE</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase">RMSE</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase">Retrains</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase">Drift Rate</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-slate-400 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map(run => (
                <tr key={`${run.dataset}_${run.strategy}_${run.model}`} className="table-row">
                  <td className="px-4 py-3">
                    <Link to={`/dataset/${run.dataset}`} className="text-white hover:text-blue-400 font-medium capitalize">
                      {run.dataset}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <span 
                      className="badge text-white"
                      style={{ backgroundColor: STRATEGY_COLORS[run.strategy] }}
                    >
                      {STRATEGY_LABELS[run.strategy]}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span 
                      className="badge"
                      style={{ backgroundColor: MODEL_COLORS[run.model], color: run.model === 'seasonal_naive' ? 'black' : 'white' }}
                    >
                      {MODEL_LABELS[run.model]}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-blue-400 font-semibold">
                    {run.metrics.mae?.toFixed(1) || 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-slate-400">
                    {run.metrics.rmse?.toFixed(1) || 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-right text-amber-400">
                    {run.metrics.retrain_total || 0}
                  </td>
                  <td className="px-4 py-3 text-right text-slate-400">
                    {run.metrics.drift_detection_rate?.toFixed(2) || '0'}%
                  </td>
                  <td className="px-4 py-3 text-center">
                    <Link 
                      to={`/run/${run.dataset}/${run.strategy}/${run.model}`}
                      className="inline-flex items-center gap-1 text-slate-400 hover:text-white transition-colors"
                    >
                      Explore <ArrowRight className="w-4 h-4" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function SummaryCard({ icon, label, value, subtext, color }) {
  return (
    <div className="metric-card">
      <div className={`${color} mb-2`}>{icon}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {subtext && <div className="text-xs text-slate-500 mt-1 truncate">{subtext}</div>}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-64 bg-dark-700 rounded animate-pulse" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 bg-dark-700 rounded-xl animate-pulse" />
        ))}
      </div>
      <div className="h-64 bg-dark-700 rounded-xl animate-pulse" />
    </div>
  )
}
