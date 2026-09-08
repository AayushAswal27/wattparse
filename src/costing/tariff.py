"""Price wasted energy using a time-of-day tariff.

The tariff shape here is a commercial ToD structure - peak surcharge in the
evening, off-peak rebate overnight - because that is the customer this
project is aimed at. The consumption data is residential UK, so the rupee
figures demonstrate the costing method rather than reproduce a real bill.

The overnight rebate matters and cuts against the story: waste at 03:00
costs less per kWh than the same waste at 19:00. Reporting it anyway is the
point - a costing head that only ever inflates the number is not useful to
someone deciding where to spend money.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DATASET_TZ = "Europe/London"


@dataclass
class ToDTariff:
    """Time-of-day tariff. Bands are (start_hour, end_hour, multiplier) in
    local time, applied to base_rate. Hours not covered by a band are billed
    at base_rate."""
    name: str = "Delhi commercial ToD (illustrative)"
    base_rate: float = 8.50            # rupees per kWh
    currency: str = "INR"
    tz: str = DATASET_TZ
    bands: list = field(default_factory=lambda: [
        (17, 23, 1.20),                # evening peak, +20%
        (23, 24, 0.80),                # night off-peak, -20%
        (0, 6, 0.80),
    ])

    def rate_at(self, ts):
        """Rupees per kWh at this instant."""
        local = ts.tz_convert(self.tz)
        hour = local.hour + local.minute / 60
        for start, end, mult in self.bands:
            if start <= hour < end:
                return self.base_rate * mult
        return self.base_rate

    def cost(self, start, end, kwh, period_s=60):
        """Cost of kwh spread evenly over a run.

        Sampled rather than using the start-hour rate: a run that crosses a
        band boundary is billed at both rates, which is how ToD actually
        works.
        """
        idx = pd.date_range(start, end, freq=f"{period_s}s", tz=start.tz)
        if len(idx) == 0:
            return kwh * self.base_rate
        per_step = kwh / len(idx)
        return float(sum(self.rate_at(ts) * per_step for ts in idx))


def price_runs(flagged, tariff):
    """Add cost columns to a flagged run table from waste.flag_runs()."""
    if flagged.empty:
        return flagged.assign(cost=None, wasted_cost=None, rate_avg=None)

    out = flagged.copy()
    out["cost"] = [
        round(tariff.cost(r.start, r.end, r.energy_kwh), 2)
        for r in out.itertuples()
    ]
    out["wasted_cost"] = [
        round(tariff.cost(r.start, r.end, r.wasted_kwh), 2)
        for r in out.itertuples()
    ]
    out["rate_avg"] = (out["cost"] / out["energy_kwh"]).round(2)
    return out


def findings_report(priced, appliance, tariff, period_days=None, top=10):
    """The artefact: ranked waste findings with money attached."""
    waste = priced[priced["outside_hours"]].copy()
    lines = []

    total_cost = float(priced["cost"].sum())
    waste_cost = float(waste["wasted_cost"].sum()) if len(waste) else 0.0
    waste_kwh = float(waste["wasted_kwh"].sum()) if len(waste) else 0.0

    lines.append(f"{appliance.upper()} - waste findings")
    lines.append(f"tariff: {tariff.name}, base {tariff.currency} "
                 f"{tariff.base_rate}/kWh")
    lines.append("")
    lines.append(f"  runs observed        {len(priced)}")
    lines.append(f"  outside hours        {len(waste)}")
    lines.append(f"  total energy         {priced['energy_kwh'].sum():.2f} kWh"
                 f"  ({tariff.currency} {total_cost:.2f})")
    lines.append(f"  wasted energy        {waste_kwh:.2f} kWh"
                 f"  ({tariff.currency} {waste_cost:.2f})")

    if period_days:
        annual = waste_cost * 365 / period_days
        lines.append(f"  annualised waste     {tariff.currency} {annual:.2f}"
                     f"  (from {period_days} days observed)")

    if len(waste):
        lines.append("")
        lines.append("  worst findings:")
        top_n = waste.sort_values("wasted_cost", ascending=False).head(top)
        for r in top_n.itertuples():
            lines.append(
                f"    {r.local_start:%Y-%m-%d %H:%M}  "
                f"{int(r.duration_s/60):3d} min  "
                f"{r.energy_kwh:5.2f} kWh  "
                f"{tariff.currency} {r.wasted_cost:6.2f}  "
                f"@ {r.rate_avg:.2f}/kWh")
    else:
        lines.append("")
        lines.append("  no out-of-hours runs detected")

    return "\n".join(lines)


def scale_projection(waste_cost, period_days, load_multiple, label="AHU"):
    """Project residential findings to a commercial load size.

    This is an assumption, not a measurement. A 15 kW AHU draws roughly 8x a
    1.8 kW dishwasher, so the same detected out-of-hours pattern costs 8x.
    Stated explicitly so nobody mistakes it for a result.
    """
    annual = waste_cost * 365 / period_days
    return (f"At {load_multiple}x load ({label}-scale), the same out-of-hours "
            f"pattern would cost approximately INR {annual * load_multiple:,.0f} "
            f"per year. This is a scaling assumption applied to residential "
            f"measurements, not a commercial measurement.")