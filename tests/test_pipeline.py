"""Tests for the pipeline, run against the synthetic store.

    pytest tests/ -v

No real data needed - make_fake_ukdale builds a 6 MB file with the same
structure. If something breaks here the bug is in the code, not in UK-DALE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.align import align
from src.data.load_ukdale import appliance_map, list_buildings, load_mains
from src.data.windows import WINDOW, Normaliser, make_windows
from src.detect.waste import OccupancySchedule, flag_runs
from src.evaluate.metrics import to_states
from src.evaluate.runs import extract_runs, run_scores
from tests.make_fake_ukdale import build

H5 = Path("data/raw/fake_ukdale.h5")


@pytest.fixture(scope="module", autouse=True)
def fake_store():
    if not H5.exists():
        build(H5)
    return H5


# --- loading -------------------------------------------------------------

def test_buildings_and_appliances():
    assert list_buildings(H5) == [1, 5]
    assert "kettle" in appliance_map(H5, 1)


def test_mains_is_tz_aware_and_sorted():
    m = load_mains(H5, 1)
    assert m.index.tz is not None
    assert m.index.is_monotonic_increasing


# --- alignment -----------------------------------------------------------

def test_align_puts_both_on_one_index():
    df = align(H5, 1, "kettle", start="2013-04-02", end="2013-04-03")
    assert list(df.columns) == ["mains", "kettle"]
    deltas = df.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1          # regular grid


def test_kettle_spike_appears_in_mains():
    """The Phase 1 gate, as an assertion."""
    df = align(H5, 1, "kettle", start="2013-04-02", end="2013-04-03")
    on = df["kettle"] > 1000
    assert on.sum() > 0
    lift = df.loc[on, "mains"].mean() - df.loc[~on, "mains"].mean()
    assert lift > 2000


# --- windowing -----------------------------------------------------------

def test_window_target_is_the_midpoint():
    """Off-by-one here silently corrupts every result downstream."""
    df = align(H5, 1, "kettle", start="2013-04-02", end="2013-04-03")
    X, y = make_windows(df, "kettle", stride=1)
    assert X.shape[1] == WINDOW
    assert len(X) == len(df) - WINDOW + 1
    half = WINDOW // 2
    expected = df["kettle"].to_numpy()[half:half + len(y)]
    assert np.allclose(y, expected)


def test_windows_with_nan_are_dropped():
    df = align(H5, 1, "kettle", start="2013-04-02", end="2013-04-03")
    clean, _ = make_windows(df, "kettle")
    df.iloc[500:505, 0] = np.nan
    dirty, _ = make_windows(df, "kettle")
    assert len(dirty) < len(clean)


def test_normaliser_does_not_refit_on_new_data():
    """Fitting on test data leaks its distribution into the numbers."""
    df = align(H5, 1, "kettle", start="2013-04-01", end="2013-04-03")
    X, y = make_windows(df, "kettle", stride=6)
    n = Normaliser().fit(X[:100], y[:100])
    Xa, _ = n.transform(X[:100], y[:100])
    Xb = n.transform(X[100:])
    assert abs(Xa.mean()) < 1e-4          # fitted set is centred
    assert abs(Xb.mean()) > 1e-6          # the rest is not


# --- states and runs -----------------------------------------------------

def test_min_on_filters_single_sample_spikes():
    idx = pd.date_range("2014-01-01", periods=100, freq="6s", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s.iloc[50] = 3000.0                   # one sample only
    assert to_states(s, 1000, min_on_s=0).sum() == 1
    assert to_states(s, 1000, min_on_s=12).sum() == 0


def test_extract_runs_finds_boundaries():
    idx = pd.date_range("2014-01-01", periods=2000, freq="6s", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s.iloc[100:200] = 2000.0              # 600 s
    s.iloc[900:1000] = 2000.0
    runs = extract_runs(s, 1000, min_duration_s=60)
    assert len(runs) == 2
    assert runs.iloc[0]["duration_s"] == 600
    assert runs.iloc[0]["energy_kwh"] == pytest.approx(2000 * 600 / 3_600_000)


def test_min_duration_removes_short_detections():
    """min_duration_s filters runs that survive min_on_s but are too short
    to be a real cycle - 20 samples is 120 s, past the 60 s min_on."""
    idx = pd.date_range("2014-01-01", periods=2000, freq="6s", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s.iloc[100:200] = 2000.0             
    s.iloc[900:920] = 2000.0              
    assert len(extract_runs(s, 1000, min_duration_s=0)) == 2
    assert len(extract_runs(s, 1000, min_duration_s=300)) == 1


def test_run_scores_counts_matches():
    idx = pd.date_range("2014-01-01", periods=2000, freq="6s", tz="UTC")
    truth = pd.Series(0.0, index=idx); truth.iloc[100:200] = 2000.0
    pred = pd.Series(0.0, index=idx); pred.iloc[105:195] = 2000.0
    a = extract_runs(truth, 1000, min_duration_s=60)
    d = extract_runs(pred, 1000, min_duration_s=60)
    s = run_scores(d, a)
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0


# --- occupancy -----------------------------------------------------------

def test_run_started_inside_hours_is_not_waste():
    idx = pd.date_range("2014-08-04 08:00", periods=1000, freq="6s", tz="UTC")
    s = pd.Series(2000.0, index=idx)      # 09:00 BST, Monday
    runs = extract_runs(s, 1000, min_duration_s=60)
    fl = flag_runs(runs, OccupancySchedule())
    assert not fl["outside_hours"].any()


def test_overnight_run_is_waste():
    idx = pd.date_range("2014-08-04 02:00", periods=1000, freq="6s", tz="UTC")
    s = pd.Series(2000.0, index=idx)      # 03:00 BST
    runs = extract_runs(s, 1000, min_duration_s=60)
    fl = flag_runs(runs, OccupancySchedule())
    assert fl["outside_hours"].all()


def test_grace_period_forgives_early_start():
    """A run seven minutes before opening is not waste."""
    idx = pd.date_range("2014-08-04 05:53", periods=100, freq="6s", tz="UTC")
    s = pd.Series(2000.0, index=idx)      # 06:53 BST, opens 07:00
    runs = extract_runs(s, 1000, min_duration_s=60)
    strict = flag_runs(runs, OccupancySchedule(grace_min=0))
    lenient = flag_runs(runs, OccupancySchedule(grace_min=15))
    assert strict["outside_hours"].all()
    assert not lenient["outside_hours"].any()