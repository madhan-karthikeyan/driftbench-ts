# DriftBench-TS — Results & Findings

## 1. Experimental Setup
- Datasets used: `electricity`, `wti`, `brent`, `traffic`
- Models compared: `naive`, `seasonal_naive`, `rf`, `lgbm`, `lstm`, `tsmixer`
- Retraining strategies: `no_retrain`, `fixed_retrain`, `adaptive_retrain`
- Drift logs are available, but the results artifacts do not include detector type, thresholds, or other detector metadata

## 2. Key Observations

### 2.1 Model Behavior Over Time
All models exhibit sustained degradation under non-stationarity.
- Error autocorrelation is consistently high, especially on drift-sensitive series, indicating that failures persist over time rather than appearing as isolated spikes.
- In `electricity`, the no-retrain LSTM performs poorly (`SMAPE 105.69`, `MAE 2173.36`), and RF is also unstable (`SMAPE 86.62`, `MAE 1811.43`).
- In `wti` and `brent`, absolute errors are much smaller, but the same temporal persistence in error remains visible.

### 2.2 Drift Detection Behavior
Evaluation of drift detection is limited by missing detector metadata.
- Drift logs are present, but the outputs do not specify detector design, thresholds, or sensitivity settings.
- The aggregated metrics show `retrain_drift_triggered_retrains = 0` in the observed runs, so the adaptive strategy does not provide evidence of effective drift-triggered retraining in the recorded summaries.
- As a result, detector behavior cannot be assessed in detail from the current artifacts.

### 2.3 Retraining Strategy Comparison
Retraining policy dominates model choice in deployment settings.
- On `electricity` LSTM:
  - `no_retrain`: `MAE 2173.36`, `SMAPE 105.69`
  - `fixed_retrain`: `MAE 820.36`, `SMAPE 37.20`
  - `adaptive_retrain`: `MAE 2122.09`, `SMAPE 105.06`
- Fixed retraining consistently achieves the lowest error across the observed runs.
- Adaptive retraining, as recorded here, performs similarly to no retraining rather than delivering a clear improvement.

### 2.4 Accuracy vs Compute Trade-off
Improved accuracy comes at a substantial compute cost.
- In the `electricity` LSTM run, fixed retraining reduces error substantially but requires much higher retraining frequency:
  - `fixed_retrain compute_proxy: 533.0`
  - `no_retrain compute_proxy: 1.0`
- The results indicate a clear accuracy-compute trade-off: frequent retraining improves predictive quality, but at significantly higher deployment cost.

### 2.5 Drift vs Error Relationship
The artifacts suggest a relationship between drift and persistent error, but do not allow a full causal alignment.
- High `error_autocorr_lag1` and large `error_max` values indicate that once performance degrades, the degradation tends to persist.
- However, the available results do not include aligned drift timestamps and error trajectories, so precise overlap between drift events and error spikes cannot be verified.
- The safest conclusion is that drift and error are consistent with one another, but direct temporal correspondence is not fully established in the stored outputs.

## 3. Quantitative Findings
- Fixed retraining on `electricity` LSTM reduces `SMAPE` from `105.69` to `37.20` and `MAE` from `2173.36` to `820.36`.
- Fixed retraining also increases compute substantially, with `retrain_total_retrains = 533` versus `1` under no retraining.
- On `electricity`, RF under no retraining remains strong in absolute error terms but is still substantially worse than the fixed-retrained LSTM.
- `wti` and `brent` exhibit much lower absolute errors than `electricity`, but still show strong temporal dependence in error behavior.
- The observed adaptive runs do not show drift-triggered retraining events in the stored summaries.

## 4. Key Insights
- Fixed retraining is the most effective strategy in the available results, consistently delivering the lowest observed error.
- The adaptive strategy does not show a measurable benefit in the recorded outputs, suggesting that drift-triggered retraining was either ineffective or not adequately captured in the logs.
- Drift induces persistent degradation rather than short-lived instability, as shown by high error autocorrelation and large error maxima.
- Dataset difficulty is heterogeneous: `electricity` is substantially harder than `wti` or `brent`.
- The best-performing strategy is not the most efficient one, so deployment choice depends on whether accuracy or compute cost is prioritized.

## 5. Limitations
- No detector metadata is stored, so the drift detection method itself cannot be evaluated directly.
- The adaptive strategy appears under-instrumented in the available summaries, limiting interpretation of its behavior.
- Exact per-timestep drift/error alignment is unavailable, so causal claims about drift triggering error spikes cannot be made.
- No plots or comparative visual summaries are present in the extracted artifacts.
- The analysis is limited to the results currently present in the repository.

## 6. Final Summary
The results show that forecasting under drift is fundamentally a deployment-policy problem, not only a model-selection problem. Across the observed runs, errors are persistent and strongly autocorrelated, indicating sustained performance degradation under non-stationarity.

Fixed retraining is the most effective policy in the available results, but it achieves this improvement at a substantial compute cost. The main unresolved question is whether drift-triggered retraining can match this accuracy more efficiently; the current artifacts do not provide enough detector-level metadata to answer that convincingly.
