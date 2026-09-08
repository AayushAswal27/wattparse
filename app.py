"""Streamlit interface for WattParse.

    streamlit run app.py

Needs models/*.pt and data/raw/ukdale.h5 on disk - neither is in git, so
this runs locally only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src.costing.tariff import ToDTariff, price_runs
from src.data.align import align
from src.data.windows import WINDOW, Normaliser, make_windows
from src.detect.waste import OccupancySchedule, flag_runs
from src.evaluate.cross_house import predict
from src.evaluate.metrics import DEFAULT_THRESHOLDS
from src.evaluate.runs import extract_runs, run_scores
from src.models.seq2point import Seq2Point
from src.run_pipeline import DETECT

H5 = "data/raw/ukdale.h5"

st.set_page_config(page_title="WattParse", layout="wide")


@st.cache_resource
def load_model(slug):
    path = Path("models/seq2point_%s.pt" % slug)
    if not path.exists():
        return None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Seq2Point(dilations=tuple(ckpt["dilations"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, Normaliser.from_dict(ckpt["normaliser"]), device


@st.cache_data(show_spinner=False)
def disaggregate(appliance, house, start, end):
    """Cached so changing the tariff or hours does not re-run the model."""
    slug = appliance.replace(" ", "_")
    loaded = load_model(slug)
    if loaded is None:
        return None
    model, norm, device = loaded

    df = align(H5, house, appliance, start=str(start), end=str(end))
    if df.empty:
        return None
    X, y = make_windows(df, appliance, stride=1)
    if len(X) == 0:
        return None

    half = WINDOW // 2
    idx = df.index[half:half + len(y)]
    return pd.DataFrame(
        {"mains": df["mains"].reindex(idx),
         "predicted": predict(model, X, norm, device),
         "actual": y},
        index=idx,
    )


st.title("WattParse")
st.caption("Disaggregating one building meter into appliance traces, then "
           "pricing what ran outside occupancy hours.")

with st.sidebar:
    st.header("Building")
    appliance = st.selectbox("Appliance", ["dish washer", "kettle", "microwave"])
    house = st.selectbox("House", [5, 1, 2, 4], help="5 is the held-out test house")

    defaults = {"dish washer": ("2014-08-01", "2014-09-01"),
                "kettle": ("2014-07-01", "2014-07-08"),
                "microwave": ("2014-08-01", "2014-09-01")}
    d0, d1 = defaults[appliance]
    start = st.date_input("From", pd.Timestamp(d0))
    end = st.date_input("To", pd.Timestamp(d1))

    st.header("Occupancy")
    open_h = st.slider("Opens", 0, 12, 7)
    close_h = st.slider("Closes", 12, 24, 23)
    grace = st.slider("Grace (min)", 0, 60, 15)

    st.header("Tariff")
    rate = st.number_input("Base rate (INR/kWh)", 1.0, 30.0, 8.50, 0.25)

if start >= end:
    st.error("End date must be after start date.")
    st.stop()

with st.spinner("Disaggregating - first run takes a minute"):
    data = disaggregate(appliance, house, start, end)

if data is None:
    st.error("No model or no data for that combination. Train the model first: "
             f'`python -m src.models.train --appliance "{appliance}"`')
    st.stop()

cfg = DETECT[appliance]
schedule = OccupancySchedule(weekday=(open_h, close_h),
                             weekend=(min(open_h + 1, 23), 24),
                             grace_min=grace)
tariff = ToDTariff(base_rate=rate)

runs = extract_runs(data["predicted"], cfg["threshold"],
                    min_duration_s=cfg["min_duration_s"])
priced = price_runs(flag_runs(runs, schedule), tariff)
waste = priced[priced["outside_hours"]] if not priced.empty else priced

days = max((end - start).days, 1)
waste_cost = float(waste["wasted_cost"].sum()) if len(waste) else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runs detected", len(priced))
c2.metric("Outside hours", len(waste))
c3.metric("Wasted energy",
          f"{float(waste['wasted_kwh'].sum()) if len(waste) else 0:.2f} kWh")
c4.metric("Annualised cost", f"INR {waste_cost * 365 / days:,.0f}")

st.subheader("Findings")
if len(waste):
    show = waste.sort_values("wasted_cost", ascending=False).copy()
    show["when"] = show["local_start"].dt.strftime("%a %d %b, %H:%M")
    show["minutes"] = (show["duration_s"] / 60).round().astype(int)
    st.dataframe(
        show[["when", "minutes", "energy_kwh", "wasted_kwh",
              "wasted_cost", "rate_avg"]]
        .rename(columns={"energy_kwh": "kWh", "wasted_kwh": "wasted kWh",
                         "wasted_cost": "INR", "rate_avg": "INR/kWh"}),
        use_container_width=True, hide_index=True)
else:
    st.info("No runs started outside occupancy hours in this window.")

st.subheader("What the model sees")
window_days = st.slider("Days to plot", 1, min(14, days), min(3, days))
plot = data.loc[data.index[0]:data.index[0] + pd.Timedelta(days=window_days)]

fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(plot.index, plot["mains"], lw=0.5, color="#4a5568")
ax[0].set_ylabel("mains (W)")
ax[1].plot(plot.index, plot["actual"], lw=0.8, color="black", label="submeter")
ax[1].plot(plot.index, plot["predicted"], lw=0.8, color="#2b6cb0",
           alpha=0.8, label="predicted")
ax[1].axhline(cfg["threshold"], ls=":", lw=0.8, color="grey")
ax[1].set_ylabel(f"{appliance} (W)")
ax[1].legend(loc="upper right", fontsize=8)

for r in waste.itertuples():
    for a in ax:
        a.axvspan(r.start, r.end, color="#e53e3e", alpha=0.25)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
st.caption("Red bands are runs flagged as outside occupancy hours.")

with st.expander("Accuracy against the appliance submeter"):
    st.write(
        "House 5 is held out - the model never saw it during training. "
        "A real deployment has no submeter to check against; that is the "
        "reason NILM exists. These numbers exist only because UK-DALE "
        "recorded both."
    )
    tcfg = DEFAULT_THRESHOLDS[appliance]
    actual_runs = extract_runs(data["actual"], tcfg["on_power"],
                               min_duration_s=cfg["min_duration_s"] // 2)
    scores = run_scores(runs, actual_runs)
    s1, s2, s3 = st.columns(3)
    s1.metric("Run precision", scores["precision"])
    s2.metric("Run recall", scores["recall"])
    s3.metric("Run F1", scores["f1"])
    st.write(f"Detected {scores['runs_detected']} runs, "
             f"{scores['runs_actual']} actually occurred: "
             f"{scores['tp']} matched, {scores['fp']} false, "
             f"{scores['fn']} missed.")
    st.write(f"Total energy: predicted "
             f"{priced['energy_kwh'].sum():.2f} kWh vs actual "
             f"{data['actual'].sum() * 6 / 3_600_000:.2f} kWh.")