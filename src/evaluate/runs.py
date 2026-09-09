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

def extend_runs(runs, mains, margin=0.5, max_extend_s=3600, period_s=6,
                context_s=10800):
    """Widen detected runs to where the aggregate stops showing the load.

    The model's prediction falls below threshold near the edges of a real run
    while staying above it in the middle, so detected runs come out short and
    their energy is understated. The aggregate meter does not have that
    problem: it is the raw signal.

    For each run, walk outward while mains stays `margin` x the run's mean
    appliance power above the local baseline.

    Baseline is a low percentile over a wide window around the run, not the
    mean of the hour before. The hour before a *detected* run is often still
    inside the real run - that is the whole problem being fixed - so an
    adjacent window is contaminated by the very load being measured.
    """
    if runs.empty:
        return runs

    out = runs.copy()
    steps = int(max_extend_s // period_s)
    ctx = int(context_s // period_s)
    vals = mains.to_numpy()
    index = mains.index

    starts, ends = [], []
    for r in out.itertuples():
        i0 = index.searchsorted(r.start)
        i1 = index.searchsorted(r.end)

        lo, hi = max(0, i0 - ctx), min(len(vals), i1 + ctx)
        baseline = float(np.percentile(vals[lo:hi], 20))
        floor = baseline + margin * r.mean_w

        j = i0
        while j > 0 and i0 - j < steps and vals[j - 1] > floor:
            j -= 1
        k = i1
        while k < len(vals) - 1 and k - i1 < steps and vals[k + 1] > floor:
            k += 1

        starts.append(index[j])
        ends.append(index[k])

    out["start"], out["end"] = starts, ends
    out["duration_s"] = [
        int((e - s).total_seconds()) + period_s for s, e in zip(starts, ends)
    ]
    # Energy over the widened span at the run's own mean power: the model
    # gives amplitude reliably, only the boundaries are wrong.
    out["energy_kwh"] = out["mean_w"] * out["duration_s"] / 3_600_000
    return out