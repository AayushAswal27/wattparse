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
from src.evaluate.runs import extend_runs, extract_runs, run_scores
from src.models.seq2point import Seq2Point
from src.run_pipeline import DETECT

H5 = "data/raw/ukdale.h5"

# Schematic-paper palette. Crimson is reserved for waste and appears in
# exactly three places: the headline figure, the plot bands, the cost column.
GROUND, INK, MUTED, RULE = "#EEF1F4", "#16202B", "#62737F", "#CBD4DB"
SIGNAL, WASTE, MAINS = "#1F6F8B", "#A21F4B", "#8FA3B0"

st.set_page_config(page_title="WattParse", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    "<style>\n"
    "@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600&display=swap');\n"
    f"""
  html, body, [class*="css"], .stApp {{
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
      font-feature-settings: 'tnum' 1;   /* every number here is compared */
  }}
  .block-container {{ padding-top: 2.6rem; max-width: 1180px; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .wp-mast {{
      display: flex; align-items: baseline; gap: .9rem;
      border-bottom: 1px solid {RULE}; padding-bottom: .55rem;
  }}
  .wp-mast h1 {{
      font-size: 1.32rem; font-weight: 600; letter-spacing: -.012em;
      margin: 0; color: {INK};
  }}
  .wp-mast span {{ font-size: .93rem; color: {MUTED}; }}

  .wp-hero {{
      display: flex; align-items: flex-end; gap: 2.6rem;
      margin: 2.1rem 0 1.9rem 0; flex-wrap: wrap;
  }}
  .wp-figure {{
      font-size: 4.1rem; font-weight: 600; line-height: .92;
      letter-spacing: -.035em; color: {WASTE}; white-space: nowrap;
  }}
  .wp-figure.quiet {{ color: {INK}; }}
  .wp-figure em {{ font-style: normal; font-size: 1.55rem; font-weight: 500; }}
  .wp-said {{
      font-size: 1.06rem; line-height: 1.5; color: {INK};
      max-width: 34em; padding-bottom: .3rem;
  }}
  .wp-said b {{ font-weight: 600; }}

  .wp-row {{
      display: flex; border-top: 1px solid {RULE};
      border-bottom: 1px solid {RULE}; margin-bottom: 2.2rem;
  }}
  .wp-cell {{ flex: 1; padding: .8rem 1.2rem .85rem 0; }}
  .wp-cell + .wp-cell {{ border-left: 1px solid {RULE}; padding-left: 1.2rem; }}
  .wp-val {{ font-size: 1.42rem; font-weight: 500; color: {INK};
             letter-spacing: -.015em; }}
  .wp-key {{ font-size: .82rem; color: {MUTED}; margin-top: .1rem; }}

  h2, h3 {{ font-size: 1.02rem !important; font-weight: 600 !important;
            color: {INK} !important; letter-spacing: -.005em; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}
  .stSlider label, .stSelectbox label, .stDateInput label,
  .stNumberInput label {{ font-size: .86rem !important; color: {MUTED}; }}
"""
    "</style>", unsafe_allow_html=True)


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


st.markdown(
    '<div class="wp-mast"><h1>WattParse</h1>'
    '<span>One building meter, split into appliances, priced against '
    'occupancy hours.</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### Building")
    appliance = st.selectbox("Appliance",
                             ["dish washer", "kettle", "microwave",
                              "washer dryer"])
    house = st.selectbox("House", [5, 1, 2, 4],
                         help="5 is held out - the model never trained on it")

    defaults = {"dish washer": ("2014-08-01", "2014-09-01"),
                "kettle": ("2014-07-01", "2014-07-08"),
                "microwave": ("2014-08-01", "2014-09-01"),
                "washer dryer": ("2014-08-01", "2014-09-01")}
    d0, d1 = defaults[appliance]
    start = st.date_input("From", pd.Timestamp(d0))
    end = st.date_input("To", pd.Timestamp(d1))

    st.markdown("#### Occupancy")
    open_h = st.slider("Opens", 0, 12, 7)
    close_h = st.slider("Closes", 12, 24, 23)
    grace = st.slider("Grace minutes", 0, 60, 15,
                      help="Tolerance either side, so a kettle boiled seven "
                           "minutes early is not reported")

    st.markdown("#### Tariff")
    rate = st.number_input("Base rate, INR per kWh", 1.0, 30.0, 8.50, 0.25)
    st.caption("Evening peak +20%, overnight −20%.")

if start >= end:
    st.error("Pick an end date after the start date.")
    st.stop()

with st.spinner("Reading the meter and splitting it out…"):
    data = disaggregate(appliance, house, start, end)

if data is None:
    st.warning(
        f"No model for {appliance} yet. Train one first:\n\n"
        f'`python -m src.models.train --appliance "{appliance}" '
        f'--epochs 10 --keep-off-ratio 20`')
    st.stop()

cfg = DETECT[appliance]
schedule = OccupancySchedule(weekday=(open_h, close_h),
                             weekend=(min(open_h + 1, 23), 24),
                             grace_min=grace)
tariff = ToDTariff(base_rate=rate)

runs = extract_runs(data["predicted"], cfg["threshold"],
                    min_duration_s=cfg["min_duration_s"])
runs = extend_runs(runs, data["mains"], margin=1.0)
priced = price_runs(flag_runs(runs, schedule), tariff)
waste = priced[priced["outside_hours"]] if not priced.empty else priced

days = max((end - start).days, 1)
waste_cost = float(waste["wasted_cost"].sum()) if len(waste) else 0.0
waste_kwh = float(waste["wasted_kwh"].sum()) if len(waste) else 0.0
annual = waste_cost * 365 / days

if len(waste):
    hour = int(waste["local_start"].dt.hour.mode().iloc[0])
    said = (f"The {appliance} started outside occupancy hours "
            f"<b>{len(waste)} time{'s' if len(waste) > 1 else ''}</b> in "
            f"{days} days, most often around {hour:02d}:00. "
            f"That is INR {waste_cost:,.2f} of electricity over the period.")
    figure_class = "wp-figure"
else:
    said = (f"Nothing started outside occupancy hours. All "
            f"{len(priced)} {appliance} runs in these {days} days began "
            f"while the building was in use.")
    figure_class = "wp-figure quiet"

st.markdown(f"""
<div class="wp-hero">
  <div>
    <div class="{figure_class}"><em>INR</em> {annual:,.0f}</div>
    <div class="wp-key" style="margin-top:.45rem">projected over a year</div>
  </div>
  <div class="wp-said">{said}</div>
</div>
<div class="wp-row">
  <div class="wp-cell"><div class="wp-val">{len(priced)}</div>
    <div class="wp-key">runs detected</div></div>
  <div class="wp-cell"><div class="wp-val">{len(waste)}</div>
    <div class="wp-key">started outside hours</div></div>
  <div class="wp-cell"><div class="wp-val">{waste_kwh:,.2f} kWh</div>
    <div class="wp-key">wasted energy</div></div>
  <div class="wp-cell"><div class="wp-val">{priced['energy_kwh'].sum():,.1f} kWh</div>
    <div class="wp-key">total for this appliance</div></div>
</div>
""", unsafe_allow_html=True)

# --- the plot: the only place you can watch the method work ---------------
st.markdown("### What the model pulls out of the meter")
window_days = st.slider("Days shown", 1, min(14, days), min(3, days),
                        label_visibility="collapsed")
plot = data.loc[data.index[0]:data.index[0] + pd.Timedelta(days=window_days)]

fig, ax = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [1, 1.25]})
fig.patch.set_facecolor(GROUND)

ax[0].plot(plot.index, plot["mains"], lw=.55, color=MAINS)
ax[0].set_ylabel("whole building (W)", fontsize=9, color=MUTED)

ax[1].plot(plot.index, plot["actual"], lw=1.0, color=INK,
           label="submeter", zorder=3)
ax[1].plot(plot.index, plot["predicted"], lw=1.0, color=SIGNAL,
           alpha=.9, label="model", zorder=2)
ax[1].set_ylabel(f"{appliance} (W)", fontsize=9, color=MUTED)
leg = ax[1].legend(loc="upper right", fontsize=9, frameon=False)
for t in leg.get_texts():
    t.set_color(MUTED)

for r in waste.itertuples():
    for a in ax:
        a.axvspan(r.start, r.end, color=WASTE, alpha=.16, lw=0, zorder=1)

for a in ax:
    a.set_facecolor(GROUND)
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(RULE)
    a.tick_params(colors=MUTED, labelsize=9, length=3)
    a.grid(axis="y", color=RULE, lw=.5, alpha=.6)
    a.set_axisbelow(True)

fig.tight_layout()
st.pyplot(fig)
st.caption("Shaded bands are runs that began outside occupancy hours."
           if len(waste) else
           "No shaded bands: nothing began outside occupancy hours.")

# --- findings -------------------------------------------------------------
st.markdown("### Findings")
if len(waste):
    show = waste.sort_values("wasted_cost", ascending=False).copy()
    show["When"] = show["local_start"].dt.strftime("%a %d %b, %H:%M")
    show["Minutes"] = (show["duration_s"] / 60).round().astype(int)
    show["Used"] = show["energy_kwh"].round(2)
    show["Wasted"] = show["wasted_kwh"].round(2)
    show["Cost"] = show["wasted_cost"].round(2)
    show["Rate"] = show["rate_avg"].round(2)
    st.dataframe(show[["When", "Minutes", "Used", "Wasted", "Cost", "Rate"]],
                 use_container_width=True, hide_index=True,
                 column_config={
                     "Used": st.column_config.NumberColumn("Used (kWh)"),
                     "Wasted": st.column_config.NumberColumn("Wasted (kWh)"),
                     "Cost": st.column_config.NumberColumn("Cost (INR)"),
                     "Rate": st.column_config.NumberColumn("INR / kWh")})
else:
    st.markdown(
        f'<div style="border:1px solid {RULE};padding:1.4rem 1.5rem;'
        f'color:{MUTED};font-size:.95rem">Nothing to report for this window. '
        f'Widen the occupancy hours in the sidebar, or pick a longer date '
        f'range, to see what would be flagged.</div>', unsafe_allow_html=True)

# --- accuracy -------------------------------------------------------------
with st.expander("How accurate is this?"):
    st.write(
        "House 5 is held out — the model never saw it during training. "
        "A real building has no submeter to check against, which is the "
        "whole reason this method exists. These numbers only exist because "
        "UK-DALE recorded both.")
    tcfg = DEFAULT_THRESHOLDS[appliance]
    actual_runs = extract_runs(data["actual"], tcfg["on_power"],
                               min_duration_s=cfg["min_duration_s"] // 2)
    s = run_scores(runs, actual_runs)
    st.markdown(f"""
<div class="wp-row" style="margin:1rem 0 .4rem 0">
  <div class="wp-cell"><div class="wp-val">{s['precision']:.2f}</div>
    <div class="wp-key">run precision</div></div>
  <div class="wp-cell"><div class="wp-val">{s['recall']:.2f}</div>
    <div class="wp-key">run recall</div></div>
  <div class="wp-cell"><div class="wp-val">{s['f1']:.2f}</div>
    <div class="wp-key">run F1</div></div>
</div>
""", unsafe_allow_html=True)
    st.write(
        f"Found {s['runs_detected']} runs where {s['runs_actual']} happened: "
        f"{s['tp']} matched, {s['fp']} were not real, {s['fn']} were missed. "
        f"Energy came out at {priced['energy_kwh'].sum():.2f} kWh against a "
        f"true {data['actual'].sum() * 6 / 3_600_000:.2f} kWh.")