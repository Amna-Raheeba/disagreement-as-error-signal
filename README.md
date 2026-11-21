# Disagreement as an Error Signal

## 1. What this is

A deterministic, offline tool that treats **disagreement between post-hoc
explainers** as a label-free, per-instance predictor of model error. For
each prediction, four stdlib explainers (leave-one-feature-out,
per-instance permutation, sampled Shapley, local surrogate) explain the
same model output. Their disagreement — in feature ranking, in which
features they name as important, and in the direction they push — is
combined into a single **distrust score**. The tool measures how well
that score predicts whether the model is actually wrong, and calibrates
a **distrust threshold** above which predictions are routed to review.

No labels are needed at inference time: the signal comes only from the
explainers arguing with each other.

## 2. The novel mechanism

The explainability literature treats explainer disagreement as a
*problem* — something to measure, report, or train away. This project
treats it as a *signal*:

1. Explain one prediction with several explainers that have different
   inductive biases (removal-based, permutation-based, game-theoretic,
   surrogate-based).
2. Compute a disagreement vector per instance:
   - **rank spread** — (1 − mean pairwise Spearman correlation) / 2
     between attribution rankings;
   - **top-k gap** — 1 − mean pairwise overlap of the top-3 features by
     absolute attribution;
   - **sign conflict** — the share of attribution weight on features
     where explainers push in opposite directions.
3. The composite distrust score is the mean of the three components.
4. Calibrate a distrust threshold on held-out data at a target review
   rate, and route everything above it to review.

When a model is on firm ground, explainers converge. When the model is
in a region it does not understand — an interaction it cannot represent,
a relationship that has flipped — the explainers split their credit
differently, and that split is visible without any label.

## 3. Prior art and the differentiator

Krishna et al. (2022) named the "disagreement problem" in explainable
ML; follow-ups (Koenen et al.'s simulation study; PEAR, which trains
models toward explainer consensus; adaptive feature-grouping work)
measure disagreement or reduce it. Confidence-based and
ensemble-variance-based uncertainty are the standard label-free error
signals.

**Differentiator:** no prior work evaluates explainer disagreement as a
per-instance predictor of ground-truth model error, and none builds a
calibrated triage rule from it. This tool does both, head-to-head
against confidence and ensemble variance, on data whose hard regions
are planted and known.

## 4. Measured headline results

Seed 7; 800 training rows; 250 validation and 250 test rows explained
per model (3 models × 4 explainers); synthetic data from a known
process (see §7 for the caveat). Full tables in `results/`.

- **Disagreement AUC (pooled): 0.535** — vs **0.682** for model
  confidence, **0.640** for ensemble variance, **0.508** for random.
  Read this honestly: on this data, plain confidence is the stronger
  single signal. Disagreement is a real but modest signal, and it is
  **model-class dependent** (next point).
- Per model, disagreement AUC: decision tree **0.595**, k-NN **0.554**,
  logistic regression **0.458** — below chance. A linear model's
  explainers agree by construction (all four recover the same linear
  form), so their disagreement measures estimation noise, not model
  failure. For the tree, mean disagreement is 0.233 on errors vs 0.205
  on correct predictions; for the linear model it is 0.058 vs 0.061 —
  no separation at all.
- By planted region (pooled over models): on **normal** rows
  disagreement AUC 0.605 vs confidence 0.724; in the **shift pocket**
  0.565 vs 0.601; in the **interaction pocket** 0.485 vs 0.497 —
  *both* signals are at chance there, because the models are
  confidently and consistently wrong and the explainers faithfully
  explain the wrong reasoning. That blind spot is a finding, not a
  footnote: disagreement detects model-form instability, not
  model-form bias.
- Rank-averaging disagreement with confidence gives pooled AUC
  **0.651**, close to confidence alone (0.682): the signals overlap
  heavily and disagreement's independent contribution is small here.
- A distrust threshold calibrated on validation data at a 20% review
  target captures **22.6%** of test errors in the review queue
  (disagreement alone) and **26.9%** when calibrated on the combined
  score.

## 5. Layout

```
disagreement_signal/   package: data, models, explainers, disagreement,
                       metrics, evaluate, report, cli
data/                  dataset.csv — the seeded dataset (all splits)
tests/                 pytest suite
notebooks/             01_demo_run.py, 02_evaluation_experiment.py
examples/              run_demo.py + demo_case.json (one explained row)
scripts/               build_cases.py, reproduce_results.py (drift guard)
results/               evaluation.json/.csv, evaluation_summary.md,
                       HEADLINE.md
app.py                 Streamlit dashboard (3 tabs)
```

## 6. How to run

```
pip install -r requirements.txt
pytest                                   # full test suite
python -m disagreement_signal demo       # explain 3 rows
python -m disagreement_signal evaluate   # regenerate results/
python scripts/reproduce_results.py      # drift check vs stored results
streamlit run app.py                     # dashboard
```

Pure Python standard library (plus pytest / Streamlit / matplotlib for
tests, the dashboard, and notebooks). No API keys, no downloads, no
network at runtime. Everything is seeded and deterministic.

## 7. Honest limits

- **Synthetic data only.** The data-generating process is known and the
  hard regions are planted by construction; the numbers measure the
  signal on this process, not on real-world or clinical data.
- **Confidence wins on this data.** The pooled AUC of disagreement
  (0.535) is below that of model confidence (0.682) and of ensemble
  variance (0.640), and combining disagreement with confidence (0.651)
  does not beat confidence alone. What the experiment establishes is
  narrower and, we think, more durable: disagreement is a genuine
  label-free error signal for nonlinear models (tree 0.595, k-NN
  0.554), it vanishes for a linear model whose explainers agree by
  construction (0.458), and it is blind — as is confidence — in the
  interaction pocket, where the model is confidently, consistently
  wrong (0.485). Anyone deploying disagreement-based triage needs to
  know all three facts.
- The models and explainers are deliberately simple stdlib versions;
  magnitudes will differ with production models and explainers.
- The distrust threshold is calibrated to a review *rate* (20%), not to
  a cost model; a deployment would price review cost against error cost.
- Region AUCs pool three models over the same rows, so model-rows are
  not independent samples.

## 8. Author and license

Author: Amna Raheeba. MIT license (see `LICENSE`).
