import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell
} from 'recharts'
import { GitCompare, Trophy, TrendingDown, Target, Zap } from 'lucide-react'
import { MODEL_COLORS, STRATEGY_COLORS, MODEL_LABELS, STRATEGY_LABELS, DATASET_COLORS } from '../types/constants'
import api from '../services/api'

export default function Compare() {
  const [comparison, setComparison] = useState(null)
  const [modelComp, setModelComp] = useState(null)
  const [strategyComp, setStrategyComp] = useState(null)
  const [robustness, setRobustness] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.getComparison(),
      api.getModelComparison(),
      api.getStrategyComparison(),
      api.getRobustness()
    ]).then(([comp, models, strategies, robust]) => {
      setComparison(comp)
      setModelComp(models)
      setStrategyComp(strategies)
      setRobustness(robust)
    }).catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingSkeleton />

  if (!comparison) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-500">No comparison data available</p>
      </div>
    )
  }

  const { heatmap, best_per_dataset, datasets, models, strategies } = comparison

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <GitCompare className="w-7 h-7 text-purple-400" />
          Global Comparison
        </h1>
        <p className="text-slate-400 mt-1">Cross-dataset analysis of models and strategies</p>
      </div>

      {/* Best Per Dataset */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center gap-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            Best Strategy + Model per Dataset
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {best_per_dataset?.map(item => (
            <div key={item.dataset} className="relative overflow-hidden rounded-xl bg-dark-700 p-4 border-l-4" style={{ borderLeftColor: DATASET_COLORS[item.dataset] }}>
              <div className="text-lg font-bold text-white capitalize mb-3">{item.dataset}</div>
              <div className="flex flex-wrap gap-2 mb-3">
                <span 
                  className="badge text-white"
                  style={{ backgroundColor: STRATEGY_COLORS[item.best_strategy] }}
                >
                  {STRATEGY_LABELS[item.best_strategy]}
                </span>
                <span 
                  className="badge"
                  style={{ backgroundColor: MODEL_COLORS[item.best_model], color: item.best_model === 'seasonal_naive' ? 'black' : 'white' }}
                >
                  {MODEL_LABELS[item.best_model]}
                </span>
              </div>
              <div className="text-3xl font-bold text-green-400">{item.best_mae.toFixed(1)}</div>
              <div className="text-xs text-slate-500">MAE • {item.improvement.toFixed(1)}% vs worst</div>
            </div>
          ))}
        </div>
      </div>

      {/* Heatmap */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title flex items-center gap-2">
            <Target className="w-5 h-5 text-blue-400" />
            Performance Heatmap (MAE by Dataset × Strategy)
          </h2>
        </div>
        <div className="overflow-x-auto">
          <HeatmapTable heatmap={heatmap} datasets={datasets} strategies={strategies} models={models} />
        </div>
        <div className="mt-4 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500" />
            <span className="text-xs text-slate-400">Best</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500" />
            <span className="text-xs text-slate-400">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-red-500" />
            <span className="text-xs text-slate-400">Worst</span>
          </div>
        </div>
      </div>

      {/* Model Comparison */}
      {modelComp && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Model Rankings</h2>
            </div>
            <div className="space-y-3">
              {modelComp.comparison?.map((item, i) => (
                <div key={item.model} className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                    i === 0 ? 'bg-amber-500 text-black' : i === 1 ? 'bg-slate-400 text-black' : i === 2 ? 'bg-amber-700 text-white' : 'bg-dark-700 text-slate-400'
                  }`}>
                    {i + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span 
                        className="badge"
                        style={{ backgroundColor: MODEL_COLORS[item.model], color: item.model === 'seasonal_naive' ? 'black' : 'white' }}
                      >
                        {MODEL_LABELS[item.model] || item.model}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-white">{item.avg_mae}</div>
                    <div className="text-xs text-slate-500">avg MAE</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Strategy Rankings</h2>
            </div>
            <div className="space-y-3">
              {strategyComp?.comparison?.map((item, i) => (
                <div key={item.strategy} className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                    i === 0 ? 'bg-green-500 text-black' : i === 1 ? 'bg-blue-500 text-white' : 'bg-red-500 text-white'
                  }`}>
                    {i + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span 
                        className="badge text-white"
                        style={{ backgroundColor: STRATEGY_COLORS[item.strategy] }}
                      >
                        {STRATEGY_LABELS[item.strategy]}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-white">{item.avg_mae}</div>
                    <div className="text-xs text-slate-500">avg MAE</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Robustness Plot */}
      {robustness && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              Efficiency Frontier: Error vs Retrain Cost
            </h2>
          </div>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  type="number" 
                  dataKey="retrain_cost" 
                  name="Retrain Cost" 
                  tick={{ fill: '#94a3b8' }}
                  label={{ value: 'Retrain Cost (compute proxy)', position: 'bottom', fill: '#94a3b8' }}
                />
                <YAxis 
                  type="number" 
                  dataKey="mae" 
                  name="MAE" 
                  tick={{ fill: '#94a3b8' }}
                  label={{ value: 'Error (MAE)', angle: -90, position: 'left', fill: '#94a3b8' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  formatter={(value, name) => [value.toFixed(2), name]}
                />
                <Legend />
                {strategies.map(strategy => (
                  <Scatter
                    key={strategy}
                    name={`${STRATEGY_LABELS[strategy]}`}
                    data={robustness.data?.filter(d => d.strategy === strategy)}
                    fill={STRATEGY_COLORS[strategy]}
                  >
                    {robustness.data?.filter(d => d.strategy === strategy).map((entry, i) => (
                      <Cell key={i} fill={DATASET_COLORS[entry.dataset]} />
                    ))}
                  </Scatter>
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-sm text-slate-500">
            Ideal solutions are in the bottom-left (low error, low cost). 
            Hover for details. Colors indicate dataset, shapes indicate strategy.
          </div>
        </div>
      )}
    </div>
  )
}

// Heatmap table component
function HeatmapTable({ heatmap, datasets, strategies, models }) {
  // Get min/max for color scaling
  const maes = heatmap.map(h => h.mae).filter(m => m > 0)
  const minMae = Math.min(...maes)
  const maxMae = Math.max(...maes)
  
  const getColor = (mae) => {
    if (mae === 0) return '#334155'
    const normalized = (mae - minMae) / (maxMae - minMae)
    if (normalized < 0.33) return '#22c55e'
    if (normalized < 0.66) return '#eab308'
    return '#ef4444'
  }

  return (
    <table className="w-full">
      <thead>
        <tr>
          <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400">Dataset</th>
          {strategies.map(s => (
            <th key={s} className="px-3 py-2 text-center text-xs font-semibold text-slate-400">
              {STRATEGY_LABELS[s]}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {datasets.map(dataset => (
          <tr key={dataset}>
            <td className="px-3 py-2 font-medium text-white capitalize" style={{ borderLeft: `3px solid ${DATASET_COLORS[dataset]}` }}>
              {dataset}
            </td>
            {strategies.map(strategy => {
              // Get best model for this dataset/strategy
              const best = heatmap
                .filter(h => h.dataset === dataset && h.strategy === strategy)
                .sort((a, b) => a.mae - b.mae)[0]
              
              return (
                <td key={strategy} className="px-3 py-2">
                  {best ? (
                    <Link 
                      to={`/run/${dataset}/${strategy}/${best.model}`}
                      className="block rounded-lg p-2 text-center hover:ring-2 hover:ring-white/20 transition-all"
                      style={{ backgroundColor: getColor(best.mae) }}
                    >
                      <div className="text-white font-bold">{best.mae.toFixed(0)}</div>
                      <div className="text-xs text-white/80">{MODEL_LABELS[best.model]}</div>
                    </Link>
                  ) : (
                    <div className="bg-dark-700 rounded-lg p-2 text-center text-slate-500">N/A</div>
                  )}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-64 bg-dark-700 rounded animate-pulse" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-32 bg-dark-700 rounded-xl animate-pulse" />
        ))}
      </div>
      <div className="h-96 bg-dark-700 rounded-xl animate-pulse" />
    </div>
  )
}
