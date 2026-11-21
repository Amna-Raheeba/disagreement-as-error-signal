"""Disagreement as an error signal.

Runs several post-hoc explainers on the same prediction, measures how much
they disagree, and tests whether that disagreement -- available at inference
time with no labels -- predicts whether the model is actually wrong.
"""

__version__ = "0.1.0"
