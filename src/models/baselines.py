"""Run NILMTK's CO and FHMM baselines and export predictions.

Runs in the `nilmtk` conda env (python 3.7, pandas 0.25), NOT the project
venv. It only produces predictions - scoring happens in .venv with
src/evaluate/metrics.py, so baselines and neural models share one scorer.

    /opt/anaconda3/envs/nilmtk/bin/python src/models/baselines.py --appliance kettle

Mains is loaded as apparent power to match src/data/load_ukdale.py. UK-DALE
mains carries apparent, active and voltage; letting NILMTK load all three
breaks disaggregation and would also mean the baselines see a different
signal than the neural models.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from nilmtk import DataSet
from nilmtk.disaggregate import CO, FHMMExact

H5 = os.path.expanduser("~/wattparse/data/raw/ukdale.h5")
OUT_DIR = os.path.expanduser("~/wattparse/data/processed")

TRAIN_HOUSES = {
    "kettle": [1, 2, 3, 4],
    "microwave": [1, 2, 4],
    "dish washer": [1, 2],
    "fridge freezer": [1],
    "washer dryer": [1],
}
TEST_HOUSE = 5

# CO and FHMM are slow. One month per training house is plenty for clustering.
TRAIN_WINDOWS = {
    1: ("2013-04-01", "2013-05-01"),
    2: ("2013-05-01", "2013-06-01"),
    3: ("2013-03-01", "2013-04-01"),
    4: ("2013-04-01", "2013-05-01"),
}
TEST_WINDOW = ("2014-07-01", "2014-07-08")


def load_training_data(appliance, sample_period):
    """Return (list of mains frames, list of appliance frames).

    partial_fit wants raw DataFrames, not meter objects, and mains must line
    up with the appliance data - so both come from the same window and get
    truncated to the same length.
    """
    mains_frames, app_frames = [], []

    for house in TRAIN_HOUSES[appliance]:
        ds = DataSet(H5)
        start, end = TRAIN_WINDOWS[house]
        ds.set_window(start=start, end=end)
        elec = ds.buildings[house].elec

        try:
            app_meter = elec[appliance]
        except KeyError:
            print("  house %d has no '%s', skipping" % (house, appliance))
            continue

        mains_df = next(elec.mains().load(
            sample_period=sample_period,
            physical_quantity="power", ac_type="apparent"))
        app_df = next(app_meter.load(sample_period=sample_period))

        n = min(len(mains_df), len(app_df))
        if n == 0:
            print("  house %d: empty window, skipping" % house)
            continue

        mains_frames.append(mains_df.iloc[:n])
        app_frames.append(app_df.iloc[:n])
        print("  house %d: %d samples" % (house, n))

    return mains_frames, app_frames


def as_series(obj, index):
    """One column on the given index.

    NILMTK's disaggregation output comes back tz-naive while the meter data
    is tz-aware, so joining them directly fails. Everything here comes from
    the same window at the same sample period in the same order, so the test
    mains index is the safe common reference.
    """
    s = obj.iloc[:, 0] if hasattr(obj, "columns") else obj
    n = min(len(s), len(index))
    return pd.Series(s.values[:n], index=index[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default="kettle")
    ap.add_argument("--sample-period", type=int, default=6)
    args = ap.parse_args()

    appliance, sp = args.appliance, args.sample_period

    print("training houses:", TRAIN_HOUSES[appliance])
    train_mains, train_app = load_training_data(appliance, sp)
    if not train_mains:
        raise SystemExit("no training data found")

    test = DataSet(H5)
    test.set_window(start=TEST_WINDOW[0], end=TEST_WINDOW[1])
    test_elec = test.buildings[TEST_HOUSE].elec
    test_mains = next(test_elec.mains().load(
        sample_period=sp,
        physical_quantity="power", ac_type="apparent"))
    truth = next(test_elec[appliance].load(sample_period=sp))
    print("test: %d samples" % len(test_mains))

    predictions = {}
    for name, model in [("CO", CO({})), ("FHMM", FHMMExact({}))]:
        print("training %s ..." % name)
        model.partial_fit(train_mains, [(appliance, train_app)])

        print("disaggregating %s ..." % name)
        out = model.disaggregate_chunk([test_mains.iloc[:, [0]]])
        predictions[name] = out[0] if isinstance(out, list) else out

    idx = test_mains.index
    frame = pd.DataFrame({
        "truth": as_series(truth, idx),
        "CO": as_series(predictions["CO"], idx),
        "FHMM": as_series(predictions["FHMM"], idx),
    })

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    path = os.path.join(OUT_DIR, "baseline_%s.csv" % appliance.replace(" ", "_"))
    frame.to_csv(path)
    print("wrote %s  (%d rows)" % (path, len(frame)))
    print(frame.describe())


if __name__ == "__main__":
    main()