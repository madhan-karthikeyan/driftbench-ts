const API_BASE = '/api';

async function fetchJSON(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });
  
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  
  return response.json();
}

export const api = {
  // Overview data
  getOverview: () => fetchJSON('/results'),
  
  // All datasets
  getDatasets: () => fetchJSON('/datasets'),
  
  // Dataset detail
  getDataset: (name) => fetchJSON(`/dataset/${name}`),
  
  // All runs for a dataset
  getDatasetRuns: (name) => fetchJSON(`/dataset/${name}/runs`),
  
  // Single run
  getRun: (dataset, strategy, model) => 
    fetchJSON(`/run/${dataset}/${strategy}/${model}`),
  
  // Run window metrics (time series)
  getRunWindows: (dataset, strategy, model) =>
    fetchJSON(`/run/${dataset}/${strategy}/${model}/windows`),
  
  // Comparison data
  getComparison: () => fetchJSON('/compare'),
  
  // Model comparison
  getModelComparison: () => fetchJSON('/compare/models'),
  
  // Strategy comparison
  getStrategyComparison: () => fetchJSON('/compare/strategies'),
  
  // Heatmap data
  getHeatmap: () => fetchJSON('/heatmap'),
  
  // Robustness data
  getRobustness: () => fetchJSON('/robustness'),
};

export default api;
