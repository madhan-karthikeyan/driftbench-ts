export const MODEL_COLORS = {
  naive: '#64748b',
  seasonal_naive: '#f59e0b',
  rf: '#8b5cf6',
  lgbm: '#06b6d4',
  lstm: '#ec4899',
  tsmixer: '#84cc16',
};

export const STRATEGY_COLORS = {
  no_retrain: '#ef4444',
  fixed_retrain: '#3b82f6',
  adaptive_retrain: '#22c55e',
};

export const STRATEGY_LINE_STYLE = {
  no_retrain: { strokeDasharray: '0' },
  fixed_retrain: { strokeDasharray: '5,5' },
  adaptive_retrain: { strokeDasharray: '10,5' },
};

export const DATASET_COLORS = {
  traffic: '#6366f1',
  electricity: '#f59e0b',
  wti: '#10b981',
  brent: '#ef4444',
};

export const MODEL_LABELS = {
  naive: 'Naive',
  seasonal_naive: 'Seasonal Naive',
  rf: 'Random Forest',
  lgbm: 'LightGBM',
  lstm: 'LSTM',
  tsmixer: 'TSMixer',
};

export const STRATEGY_LABELS = {
  no_retrain: 'No Retrain',
  fixed_retrain: 'Fixed Retrain',
  adaptive_retrain: 'Adaptive',
};
