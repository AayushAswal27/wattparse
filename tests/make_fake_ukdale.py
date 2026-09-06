"""
make_fake_ukdale.py
===================

Build a tiny HDF5 with the *same structure* as ukdale.h5 (node layout,
MultiIndex columns, tz-aware index, NILMTK metadata in node attrs) so you can
develop and test the loader while the real 5.4 GB download runs.

    python tests/make_fake_ukdale.py data/raw/fake_ukdale.h5

Two buildings:
  building1 - mains (apparent power, 6s) + kettle, fridge, washing machine
  building5 - mains + kettle, fridge          (deliberately fewer appliances,
                                               so appliance_availability() has
                                               something real to report)

The mains trace is literally the sum of the submeters plus a noise floor, so
`02_alignment_check.ipynb` should show kettle spikes lining up perfectly.
If your alignment code cannot pass the gate on THIS file, the bug is in your
code, not in UK-DALE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tables

TZ = "UTC"
PERIOD_S = 6
N_DAYS = 3
RNG = np.random.default_rng(27)


def _index(start: str, days: int, period_s: int) -> pd.DatetimeIndex:
    n = int(days * 24 * 3600 / period_s)
    return pd.date_range(start=start, periods=n, freq=f"{period_s}s", tz=TZ)


def _kettle(idx: pd.DatetimeIndex) -> np.ndarray:
    """~3 kW rectangular bursts of 2-4 minutes, a few times a day."""
    x = np.zeros(len(idx))
    per_day = int(24 * 3600 / PERIOD_S)
    for day_start in range(0, len(idx), per_day):
        for hour in (7, 11, 16, 21):
            start = day_start + int(hour * 3600 / PERIOD_S) + RNG.integers(-60, 60)
            dur = RNG.integers(20, 40)          # 20-40 samples = 2-4 min
            if 0 <= start < len(idx) - dur:
                x[start : start + dur] = RNG.normal(2900, 60, dur)
    return np.clip(x, 0, None)


def _fridge(idx: pd.DatetimeIndex) -> np.ndarray:
    """~90 W compressor cycling roughly 20 min on / 40 min off."""
    x = np.zeros(len(idx))
    on = int(20 * 60 / PERIOD_S)
    cycle = int(60 * 60 / PERIOD_S)
    for start in range(0, len(idx), cycle):
        x[start : start + on] = RNG.normal(90, 5, len(x[start : start + on]))
    return np.clip(x, 0, None)


def _washer(idx: pd.DatetimeIndex) -> np.ndarray:
    """Two ~90 min runs with a 2 kW heating phase then a low tumble phase."""
    x = np.zeros(len(idx))
    per_day = int(24 * 3600 / PERIOD_S)
    for day_start in range(0, len(idx), per_day):
        start = day_start + int(10 * 3600 / PERIOD_S)
        heat = int(20 * 60 / PERIOD_S)
        tumble = int(70 * 60 / PERIOD_S)
        if start + heat + tumble < len(idx):
            x[start : start + heat] = RNG.normal(2000, 80, heat)
            x[start + heat : start + heat + tumble] = RNG.normal(180, 30, tumble)
    return np.clip(x, 0, None)


def _write_meter(store: pd.HDFStore, building: int, meter: int,
                 idx: pd.DatetimeIndex, watts: np.ndarray, ac_type: str) -> None:
    cols = pd.MultiIndex.from_tuples(
        [("power", ac_type)], names=["physical_quantity", "type"]
    )
    df = pd.DataFrame(watts.reshape(-1, 1), index=idx, columns=cols)
    store.put(f"/building{building}/elec/meter{meter}", df, format="table")


def _attach_metadata(path: Path, building: int, meta: dict) -> None:
    with tables.open_file(str(path), mode="a") as h5:
        node = h5.get_node(f"/building{building}")
        node._v_attrs.metadata = meta


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    idx = _index("2013-04-01", N_DAYS, PERIOD_S)

    houses = {
        1: {"kettle": _kettle(idx), "fridge": _fridge(idx),
            "washing machine": _washer(idx)},
        5: {"kettle": _kettle(idx), "fridge": _fridge(idx)},
    }

    with pd.HDFStore(str(out), mode="w") as store:
        for b, appliances in houses.items():
            total = sum(appliances.values()) + RNG.normal(120, 15, len(idx))
            _write_meter(store, b, 1, idx, np.clip(total, 0, None), "apparent")
            for i, (name, watts) in enumerate(appliances.items(), start=2):
                _write_meter(store, b, i, idx, watts, "active")

    for b, appliances in houses.items():
        elec_meters = {1: {"site_meter": True, "device_model": "fake"}}
        appliance_meta = []
        for i, name in enumerate(appliances, start=2):
            elec_meters[i] = {"submeter_of": 1, "device_model": "fake"}
            appliance_meta.append({"type": name, "instance": 1, "meters": [i]})
        _attach_metadata(
            out, b,
            {"instance": b, "elec_meters": elec_meters,
             "appliances": appliance_meta, "timezone": "Europe/London"},
        )

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/fake_ukdale.h5"))
