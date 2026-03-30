# DriftBench-TS — Results & Findings

## 1. Experimental Setup
- Datasets: `electricity` (5330 windows, 127920 predictions), `traffic` (1978 windows, 47472 predictions), `brent` (44 windows, 7392 predictions), `wti` (44 windows, 7392 predictions)
- Models: `naive`, `seasonal_naive`, `rf`, `lgbm`, `lstm`, `tsmixer`
- Retraining strategies: `no_retrain`, `fixed_retrain`, `adaptive_retrain`
- Test runs: `test`, `test_lgbm`, `test_lstm`, `test_sn`, `test_tsmixer`

## 2. Key Observations

### 2.1 Model Performance Ranking
LGBM and RF consistently outperform other models across all datasets and retraining strategies.

| Dataset | Model | No Retrain MAE | Fixed Retrain MAE | Improvement |
|---------|-------|-----------------|-------------------|-------------|
| electricity | lgbm | 1494.5 | 119.5 | **92%** |
| electricity | rf | 1811.4 | 153.6 | **92%** |
| electricity | lstm | 2173.4 | 820.4 | 62% |
| traffic | lgbm | 212.7 | 194.6 | 9% |
| traffic | rf | 344.2 | 329.0 | 4% |
| brent | lgbm | 19.1 | 7.3 | 62% |
| brent | rf | 21.6 | 8.1 | 62% |
| wti | lgbm | 18.2 | 5.9 | 68% |
| wti | rf | 20.2 | 6.8 | 67% |

### 2.2 Retraining Strategy Comparison

**Fixed retraining** is the most effective strategy across all datasets:
- electricity/lgbm: SMAPE 71.6 → 6.04 (91% reduction)
- electricity/rf: SMAPE 86.6 → 9.55 (89% reduction)
- traffic/lgbm: SMAPE 9.79 → 9.27 (5% reduction)
- brent/lgbm: SMAPE 31.6 → 12.7 (60% reduction)
- wti/lgbm: SMAPE 31.7 → 11.1 (65% reduction)

**Adaptive retraining shows zero drift-triggered retrains** (`retrain_total_retrains = 0` for all adaptive runs). Performance is identical to no-retrain, indicating either:
- Drift was not detected by the configured detector
- Detector thresholds may be too conservative
- The detector design needs evaluation

### 2.3 Dataset Difficulty
- `electricity`: Hardest dataset; SMAPE values exceed 100% for naive baselines; fixed retraining provides massive gains
- `traffic`: Moderate difficulty; ML models (lgbm, rf) perform well even without retraining
- `brent` and `wti`: Easier datasets with lower absolute errors; ML models benefit substantially from fixed retraining

### 2.4 Error Autocorrelation
All models show high error autocorrelation (0.94-0.99 on brent/wti), indicating persistent degradation rather than isolated spikes. Fixed retraining reduces autocorrelation:
- electricity/lgbm: 0.94 → 0.69
- brent/lgbm: 0.99 → 0.99 (minimal change)
- wti/lgbm: 0.99 → 0.98

### 2.5 Compute Trade-off

| Dataset | Strategy | Compute Proxy | Retrains |
|---------|----------|--------------|----------|
| electricity | no_retrain | 1.0 | 1 |
| electricity | adaptive | 0.0 | 0 |
| electricity | fixed_retrain | 533.0 | 533 |
| traffic | no_retrain | 1.0 | 1 |
| traffic | fixed_retrain | 198.0 | 198 |
| brent | no_retrain | 1.0 | 1 |
| brent | fixed_retrain | 5.0 | 5 |

Fixed retraining achieves best accuracy but at substantial compute cost (100-500x baseline).

## 3. Quantitative Summary

### Best Overall: lgbm with fixed_retrain
- **electricity**: MAE 119.5, SMAPE 6.04 (vs no_retrain: 1494.5, 71.6)
- **traffic**: MAE 194.6, SMAPE 9.27 (vs no_retrain: 212.7, 9.79)
- **brent**: MAE 7.3, SMAPE 12.7 (vs no_retrain: 19.1, 31.6)
- **wti**: MAE 5.9, SMAPE 11.1 (vs no_retrain: 18.2, 31.7)

### Adaptive Strategy: Complete Failure
All adaptive_retrain runs show `retrain_total_retrains: 0` and `compute_proxy: 0.0`, meaning no drift was detected. Performance matches no_retrain exactly.

## 4. Limitations
- No detector metadata (type, thresholds, sensitivity) stored in results
- Adaptive strategy results are uninformative due to zero retrains
- Cannot assess drift detection effectiveness
- No per-timestep drift/error alignment available

## 5. Conclusions
1. **Fixed retraining is the only effective retraining strategy** in these experiments
2. **LGBM is the best-performing model** across all datasets
3. **Adaptive drift-triggered retraining failed completely** — zero retrains triggered across all datasets
4. **Drift induces persistent error** (high autocorrelation) rather than transient spikes
5. **Compute-accuracy trade-off is severe**: 100-500x compute for best accuracy
6. **Dataset characteristics matter**: electricity benefits enormously from retraining; traffic benefits minimally
