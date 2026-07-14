# Cross-year ranking — data

## Fairness protocol
- Same dataset for all three model families.
- Same beach overlap (train 2022 ∩ test 2025 summer).
- Same train rows, same test rows.
- Same Optuna budget: 30 trials per (model, horizon).
- Same validation window: 2025-06-01 → 2025-08-31.
- Same metric: P90-normalised relMAE on daytime hours.

## Cells
- Train: cache_2022 (full year, daytime 8-20) — 23,008 rows
- Test:  django_2025 2025-06-01 → 2025-08-31 — 4,724 rows
- Beaches: 10

| model   | horizon   |   n_rows |   relMAE_overall |   relMAE_season |   relMAE_summer |     MAE |   n_seeds |   elapsed_s | dataset   |
|:--------|:----------|---------:|-----------------:|----------------:|----------------:|--------:|----------:|------------:|:----------|
| xgb     | 10d       |     4200 |           0.3764 |          0.3764 |          0.3764 | 29.8308 |         1 |     70.1000 | data      |
| lstm    | 10d       |      520 |           0.3825 |          0.3825 |          0.3825 | 35.8954 |         1 |    129.9000 | data      |
| tft     | 10d       |      520 |           0.4040 |          0.4040 |          0.4040 | 35.7546 |         1 |   3388.1000 | data      |
| xgb     | 15d       |     3940 |           0.3791 |          0.3791 |          0.3791 | 29.9656 |         1 |     85.6000 | data      |
| lstm    | 15d       |      780 |           0.4456 |          0.4456 |          0.4456 | 39.8029 |         1 |    148.5000 | data      |
| tft     | 15d       |      780 |           0.4740 |          0.4740 |          0.4740 | 42.5464 |         1 |   3895.7000 | data      |
| xgb     | 3d        |     4564 |           0.3617 |          0.3617 |          0.3617 | 28.6599 |         1 |     71.5000 | data      |
| tft     | 3d        |      156 |           0.4145 |          0.4145 |          0.4145 | 39.5235 |         1 |   2601.4000 | data      |
| lstm    | 3d        |      156 |           0.4507 |          0.4507 |          0.4507 | 40.3723 |         1 |    130.0000 | data      |