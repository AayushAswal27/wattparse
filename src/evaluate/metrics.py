"""Scoring for disaggregation. Every model in this project is scored here.

Baselines and neural models must go through the same functions, or the
comparison measures implementation differences instead of algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Power below which an appliance counts as off, and the shortest run we treat
# as real. Both matter: threshold sets what counts as an event, min_on kills
# single-sample flicker that would otherwise dominate the false positives.
DEFAULT_THRESHOLDS = {
    "kettle": {"on_power": 1000, "min_on_s": 12, "min_off_s": 0},
    "microwave": {"on_power": 200, "min_on_s": 12, "min_off_s": 30},
    "dish washer": {"on_power": 10, "min_on_s": 1800, "min_off_s": 1800},
    "fridge freezer": {"on_power": 50, "min_on_s": 60, "min_off_s": 12},
    "washer dryer": {"on_power": 20, "min_on_s": 1800, "min_off_s": 160},
}


def to_states(power, on_power, min_on_s=0, min_off_s=0, period_s=6):
    """Turn a power trace into a boolean ON/OFF series.

    Threshold, then close short OFF gaps, then drop short ON runs. Order
    matters - closing gaps first means a run broken by one dropped sample
    stays one run instead of becoming two runs too short to survive.
    """
    state = (power > on_power).fillna(False)

    if min_off_s:
        state = _close_short_runs(state, False, int(min_off_s // period_s))
    if min_on_s:
        state = _close_short_runs(state, True, int(min_on_s // period_s))
    return state


def _close_short_runs(state, value, min_len):
    """Flip runs of `value` shorter than min_len to the opposite value."""
    if min_len <= 1:
        return state
    out = state.copy()
    groups = (state != state.shift()).cumsum()
    for _, idx in state.groupby(groups).groups.items():
        if state.loc[idx[0]] == value and len(idx) < min_len:
            out.loc[idx] = not value
    return out


def mae(pred, truth):
    """Mean absolute error in watts. Dominated by the off periods, since most
    appliances are off most of the time - low MAE alone means little."""
    d = pd.concat([pred, truth], axis=1).dropna()
    return float((d.iloc[:, 0] - d.iloc[:, 1]).abs().mean())


def sae(pred, truth):
    """Signal aggregate error: relative error in total energy over the window.
    Near zero can hide a model that gets the timing completely wrong."""
    d = pd.concat([pred, truth], axis=1).dropna()
    total_truth = d.iloc[:, 1].sum()
    if total_truth == 0:
        return float("nan")
    return float(abs(d.iloc[:, 0].sum() - total_truth) / total_truth)


def state_scores(pred, truth, on_power, min_on_s=0, min_off_s=0, period_s=6):
    """Precision, recall and F1 on ON/OFF states - the metric that matters
    for waste detection, where 'did it run' beats 'how many watts'."""
    d = pd.concat([pred.rename("p"), truth.rename("t")], axis=1).dropna()
    if d.empty:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                "tp": 0, "fp": 0, "fn": 0}

    p = to_states(d["p"], on_power, min_on_s, min_off_s, period_s)
    t = to_states(d["t"], on_power, min_on_s, min_off_s, period_s)

    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def score(pred, truth, appliance, period_s=6, thresholds=None):
    """All metrics for one appliance. This is what every experiment calls."""
    cfg = (thresholds or DEFAULT_THRESHOLDS).get(appliance)
    if cfg is None:
        raise KeyError(f"no threshold configured for '{appliance}'")

    out = {"appliance": appliance, "on_power": cfg["on_power"],
           "mae": mae(pred, truth), "sae": sae(pred, truth)}
    out.update(state_scores(pred, truth, cfg["on_power"],
                            cfg["min_on_s"], cfg["min_off_s"], period_s))
    return out


def compare(predictions, truth, appliance, period_s=6):
    """Score several models against the same ground truth.

    predictions: {"CO": series, "FHMM": series, "seq2point": series}
    """
    rows = []
    for name, pred in predictions.items():
        row = score(pred, truth, appliance, period_s)
        row["model"] = name
        rows.append(row)
    cols = ["model", "mae", "sae", "precision", "recall", "f1", "tp", "fp", "fn"]
    return pd.DataFrame(rows)[cols].set_index("model")