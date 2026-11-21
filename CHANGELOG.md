# Changelog

## 0.1.0 — 2026-11-21

- Initial release.
- Four stdlib explainers (LOCO, per-instance permutation, sampled Shapley,
  local surrogate) over three stdlib models (logistic regression, decision
  tree, k-NN).
- Disagreement vector: rank spread, top-k gap, sign conflict, and the
  composite distrust score.
- Headline experiment on seeded synthetic data with planted interaction
  and shift pockets: disagreement AUC vs confidence, ensemble variance,
  and random baselines, broken down by region.
- Distrust-threshold calibration (validation quantile at a 20% review
  target) and triage statistics.
- Streamlit dashboard with three tabs.
