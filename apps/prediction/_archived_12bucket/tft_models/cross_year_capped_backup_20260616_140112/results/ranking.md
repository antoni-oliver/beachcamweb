# Cross-year ranking — backup

## Fairness protocol
- Same dataset for all three model families.
- Same beach overlap (train 2022 ∩ test 2025 summer).
- Same train rows, same test rows.
- Same Optuna budget: 50 trials per (model, horizon).
- Same validation window: 2025-06-01 → 2025-08-31.
- Same metric: P90-normalised relMAE on daytime hours.

## Cells
- Train: cache_2022 (full year, daytime 8-20) — 9,062 rows
- Test:  django_2025 2025-06-01 → 2025-08-31 — 4,724 rows
- Beaches: 4

| model   | horizon   |   n_rows |   relMAE_overall |   relMAE_season |   relMAE_summer |     MAE |   n_seeds |   elapsed_s | dataset   |
|:--------|:----------|---------:|-----------------:|----------------:|----------------:|--------:|----------:|------------:|:----------|
| xgb     | 10d       |     4240 |           0.4077 |          0.4077 |          0.4077 | 32.3164 |         1 |     34.0000 | backup    |
| xgb     | 15d       |     4000 |           0.3975 |          0.3975 |          0.3975 | 31.8184 |         1 |     33.3000 | backup    |
| xgb     | 3d        |     4576 |           0.3859 |          0.3859 |          0.3859 | 30.3666 |         1 |     35.9000 | backup    |