"""Turn per-sample ON/OFF states into discrete appliance runs.

A run is one continuous period the appliance was on: start, end, duration,
and energy. This is the unit a facility manager thinks in - "the dishwasher
ran at 2am for 90 minutes" - and the unit Phase 5 prices.

Run-level accuracy is not sample-level F1. Scattered false-positive samples
collapse into few runs once a minimum duration is required, so a model with
poor sample F1 can still produce a usable run list. The opposite is also
possible, which is why detected runs are matched against actual ones here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.metrics import DEFAULT_THRESHOLDS, to_states


def extract_runs(power, threshold, min_on_s=60, min_off_s=60, period_s=6,
                 min_duration_s=0):
    """Return a DataFrame of runs: start, end, duration_s, energy_kwh, peak_w.

    min_duration_s drops runs shorter than a plausible cycle. Set it from the
    appliance, not from the data: a dishwasher cycle is not 30 seconds long,
    so a 30-second detection is noise whatever the model says.
    """
    state = to_states(power, threshold, min_on_s, min_off_s, period_s)
    if not state.any():
        return pd.DataFrame(columns=["start", "end", "duration_s",
                                     "energy_kwh", "peak_w", "mean_w"])

    # Boundaries are where the state changes; group consecutive True runs.
    groups = (state != state.shift()).cumsum()
    rows = []
    for _, idx in state[state].groupby(groups[state]).groups.items():
        seg = power.loc[idx]
        duration = len(idx) * period_s
        if duration < min_duration_s:
            continue
        rows.append({
            "start": idx[0],
            "end": idx[-1],
            "duration_s": duration,
            "energy_kwh": float(seg.sum()) * period_s / 3_600_000,
            "peak_w": float(seg.max()),
            "mean_w": float(seg.mean()),
        })
    return pd.DataFrame(rows)


def match_runs(detected, actual, tolerance_s=300):
    """Match detected runs to actual ones by overlap.

    A detection counts if it overlaps a real run at all, allowing a tolerance
    at each edge - a model that catches the middle of a cycle but misses the
    first two minutes has still found the run.

    Returns (matched, false_positives, missed) as DataFrames.
    """
    if detected.empty:
        return detected, detected, actual
    if actual.empty:
        return actual, detected, actual

    tol = pd.Timedelta(seconds=tolerance_s)
    matched_idx, used_actual = [], set()

    for i, d in detected.iterrows():
        for j, a in actual.iterrows():
            if j in used_actual:
                continue
            if d["start"] <= a["end"] + tol and a["start"] <= d["end"] + tol:
                matched_idx.append(i)
                used_actual.add(j)
                break

    matched = detected.loc[matched_idx]
    false_pos = detected.drop(index=matched_idx)
    missed = actual.drop(index=list(used_actual))
    return matched, false_pos, missed


def run_scores(detected, actual, tolerance_s=300):
    """Run-level precision, recall, F1 - the numbers that describe the
    findings a user would actually see."""
    matched, fp, missed = match_runs(detected, actual, tolerance_s)
    tp, n_fp, n_fn = len(matched), len(fp), len(missed)
    prec = tp / (tp + n_fp) if tp + n_fp else 0.0
    rec = tp / (tp + n_fn) if tp + n_fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"runs_detected": len(detected), "runs_actual": len(actual),
            "tp": tp, "fp": n_fp, "fn": n_fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3)}