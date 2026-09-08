"""Flag appliance runs that fall outside occupancy hours.

Occupancy is declared, not inferred. A facility manager knows their opening
hours; guessing them and being wrong means telling someone their equipment
ran overnight when it did not. Inference belongs alongside this as a check,
not underneath it as the source of truth.

UK-DALE is residential, so "occupancy" here means the household's waking
routine. The commercial analogue is opening hours; the logic is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DATASET_TZ = "Europe/London"


@dataclass
class OccupancySchedule:
    """Hours the building is expected to be in use, in LOCAL wall-clock time.

    Timestamps in this project are UTC. August in the UK is BST, so 03:00 UTC
    is 04:00 local - running this on UTC would shift every finding by an hour.
    """
    weekday: tuple[int, int] = (7, 23)      # 07:00 to 23:00
    weekend: tuple[int, int] = (8, 24)
    tz: str = DATASET_TZ
    holidays: set = field(default_factory=set)   # dates with no occupancy

    def window_for(self, ts):
        """(start_hour, end_hour) for the day this timestamp falls on."""
        if ts.date() in self.holidays:
            return None
        return self.weekend if ts.weekday() >= 5 else self.weekday

    def is_occupied(self, ts):
        """Is this single instant inside occupancy hours?"""
        local = ts.tz_convert(self.tz)
        w = self.window_for(local)
        if w is None:
            return False
        hour = local.hour + local.minute / 60
        return w[0] <= hour < w[1]

    def occupied_fraction(self, start, end, period_s=60):
        """Fraction of a run that falls inside occupancy hours.

        Sampled minute by minute rather than compared at the edges, because a
        run can straddle a boundary or span midnight into a different regime.
        """
        idx = pd.date_range(start, end, freq=f"{period_s}s", tz=start.tz)
        if len(idx) == 0:
            return 0.0
        local = idx.tz_convert(self.tz)
        inside = 0
        for ts in local:
            w = self.window_for(ts)
            if w is None:
                continue
            hour = ts.hour + ts.minute / 60
            if w[0] <= hour < w[1]:
                inside += 1
        return inside / len(idx)


def flag_runs(runs, schedule, waste_threshold=0.5):
    """Add occupancy columns to a run table from extract_runs().

    A run is waste only if it STARTED outside occupancy hours and mostly
    stayed there. Start time is the primary test: a dishwasher started at
    22:52 was started by someone who was awake, even if most of the cycle
    runs past 23:00. Equipment still running at 03:00 was not.

    waste_threshold: of the runs that started outside hours, this fraction
    must also fall outside for it to count.
    """
    if runs.empty:
        return runs.assign(local_start=None, occupied_fraction=None,
                           started_outside=None, outside_hours=None,
                           wasted_kwh=None)

    out = runs.copy()
    out["local_start"] = out["start"].dt.tz_convert(schedule.tz)
    out["occupied_fraction"] = [
        round(schedule.occupied_fraction(r.start, r.end), 3)
        for r in out.itertuples()
    ]
    out["started_outside"] = [
        not schedule.is_occupied(r.start) for r in out.itertuples()
    ]
    out["outside_hours"] = out["started_outside"] & (
        out["occupied_fraction"] <= (1 - waste_threshold))
    out["wasted_kwh"] = out["energy_kwh"] * (1 - out["occupied_fraction"])
    return out


def waste_summary(flagged, appliance="appliance"):
    """One-line-per-appliance rollup of what was wasted."""
    if flagged.empty:
        return {"appliance": appliance, "runs": 0, "waste_runs": 0,
                "total_kwh": 0.0, "wasted_kwh": 0.0}
    waste = flagged[flagged["outside_hours"]]
    return {
        "appliance": appliance,
        "runs": len(flagged),
        "waste_runs": len(waste),
        "total_kwh": round(float(flagged["energy_kwh"].sum()), 3),
        "wasted_kwh": round(float(waste["wasted_kwh"].sum()), 3),
        "worst_hour": (int(waste["local_start"].dt.hour.mode().iloc[0])
                       if len(waste) else None),
    }


def findings_table(flagged, appliance, top=10):
    """The runs a person would actually be shown, worst first."""
    waste = flagged[flagged["outside_hours"]].copy()
    if waste.empty:
        return waste
    waste = waste.sort_values("wasted_kwh", ascending=False).head(top)
    waste["appliance"] = appliance
    waste["duration_min"] = (waste["duration_s"] / 60).round(0).astype(int)
    return waste[["appliance", "local_start", "duration_min",
                  "peak_w", "energy_kwh", "wasted_kwh", "occupied_fraction"]]