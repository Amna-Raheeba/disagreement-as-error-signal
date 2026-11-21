# Headline results

- Explainer disagreement predicts model error with pooled AUC 0.535, vs 0.682 for model confidence, 0.640 for ensemble variance, and 0.508 for a random score (seed 7, synthetic data from a known process). The signal is model-class dependent: it is real for the decision tree, weak for k-NN, and absent for the linear model, whose explainers agree by construction.
- Rank-averaging disagreement with confidence gives pooled AUC 0.651, close to confidence alone (0.682) and above disagreement alone: the two signals overlap heavily, and disagreement's independent contribution is small on this data.
- A distrust threshold calibrated on validation data (20% review target) captures 0.226 of test errors in the review queue.
- All numbers are measured on seeded synthetic data with planted hard regions; they are not clinical or real-world results.
