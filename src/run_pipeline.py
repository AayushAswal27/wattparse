"""End to end: raw meter -> disaggregation -> runs -> waste -> rupees.

    python -m src.run_pipeline --appliance "dish washer" --house 5 \
        --start 2014-08-01 --end 2014-09-01

Everything downstream of the model is deterministic, so this is the whole
system in one command. Pass --truth to score the findings against the
appliance submeter, which only exists in a dataset like UK-DALE - a real
deployment has no ground truth, which is the entire reason NILM exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.costing.tariff import ToDTariff, findings_report, price_runs
from src.data.align import align
from src.data.windows import WINDOW, Normaliser, make_windows
from src.detect.waste import OccupancySchedule, flag_runs
from src.evaluate.cross_house import predict
from src.evaluate.metrics import DEFAULT_THRESHOLDS
from src.evaluate.runs import extend_runs, extract_runs, run_scores
from src.models.seq2point import Seq2Point

H5 = "data/raw/ukdale.h5"
REPORT_DIR = Path("reports")

# Decision threshold and minimum run length per appliance. Thresholds were
# selected by sweeping on a training house (see RESULTS.md); minimum
# durations are domain constraints - a dishwasher cycle is not 5 minutes.
DETECT = {
    "kettle": {"threshold": 1550, "min_duration_s": 60},
    "microwave": {"threshold": 500, "min_duration_s": 30},
    "dish washer": {"threshold": 100, "min_duration_s": 1200},
    "fridge freezer": {"threshold": 50, "min_duration_s": 300},
    "washer dryer": {"threshold": 100, "min_duration_s": 1200},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default="dish washer")
    ap.add_argument("--house", type=int, default=5)
    ap.add_argument("--start", default="2014-08-01")
    ap.add_argument("--end", default="2014-09-01")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--open-hour", type=float, default=7)
    ap.add_argument("--close-hour", type=float, default=23)
    ap.add_argument("--rate", type=float, default=8.50)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="run extension: multiples of appliance power above "
                         "baseline that mains must hold")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--truth", action="store_true",
                    help="also score against the appliance submeter")
    args = ap.parse_args()

    appliance = args.appliance
    slug = appliance.replace(" ", "_")
    cfg = DETECT[appliance]
    ckpt_path = Path(args.checkpoint or ("models/seq2point_%s.pt" % slug))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Seq2Point(dilations=tuple(ckpt["dilations"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    norm = Normaliser.from_dict(ckpt["normaliser"])

    print("house %d, %s to %s" % (args.house, args.start, args.end))
    print("model %s, threshold %d W, min run %d s"
          % (ckpt_path.name, cfg["threshold"], cfg["min_duration_s"]))

    df = align(args.h5, args.house, appliance, start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no data in that window")
    X, y = make_windows(df, appliance, stride=1)
    half = WINDOW // 2
    idx = df.index[half:half + len(y)]
    print("  %d samples disaggregated" % len(idx))

    pred = pd.Series(predict(model, X, norm, device), index=idx)

    runs = extract_runs(pred, cfg["threshold"],
                        min_duration_s=cfg["min_duration_s"])
    if not args.no_extend:
        # The model's output dips below threshold at the edges of a real run
        # while holding above it in the middle, so detected runs are short and
        # their energy is understated. The raw aggregate has no such problem.
        runs = extend_runs(runs, df["mains"].reindex(idx), margin=args.margin)

    schedule = OccupancySchedule(weekday=(args.open_hour, args.close_hour),
                                 weekend=(args.open_hour + 1, 24))
    tariff = ToDTariff(base_rate=args.rate)
    priced = price_runs(flag_runs(runs, schedule), tariff)

    days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days
    report = findings_report(priced, appliance, tariff, period_days=days)

    print()
    print(report)

    if args.truth:
        tcfg = DEFAULT_THRESHOLDS[appliance]
        truth = pd.Series(y, index=idx)
        actual = extract_runs(truth, tcfg["on_power"],
                              min_duration_s=cfg["min_duration_s"] // 2)
        print()
        print("run-level accuracy vs submeter:", run_scores(runs, actual))
        print("energy: predicted %.2f kWh vs actual %.2f kWh"
              % (priced["energy_kwh"].sum(), actual["energy_kwh"].sum()))

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / ("%s_house%d_%s.txt" % (slug, args.house, args.start[:7]))
    out.write_text(report + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()