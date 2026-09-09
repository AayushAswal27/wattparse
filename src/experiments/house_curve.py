"""How many training buildings does this method need?

    python -m src.experiments.house_curve --appliance kettle

Trains on 1, 2, 3 then 4 houses and evaluates each on the held-out house.
The washer dryer result made this the central question: it reached F1 0.960
on its single training house and 0.233 on house 5, the largest transfer gap
in the project, despite having the most favourable power, duration and event
density of any appliance. One training house appears to matter more.

For a commercial deployment this is the number that decides feasibility:
how many buildings must be sub-metered before the model works on an
unmetered one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluate.metrics import score
from src.models.train import APPLIANCE_HOUSES

RESULTS = Path("data/processed/house_curve.json")

# Rough data volume order - house 1 is by far the longest record.
VOLUME_ORDER = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4}


def subsets(houses, test_house):
    """One nested subset per size, not every combination.

    All 15 subsets of 4 houses would take hours. Ordering by data volume
    keeps each step a superset of the last, so the curve isolates the effect
    of adding a house rather than mixing it with which houses were picked.
    """
    pool = [h for h in houses if h != test_house]
    order = sorted(pool, key=lambda h: VOLUME_ORDER.get(h, 9))
    return [order[:n] for n in range(1, len(order) + 1)]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit("command failed")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default="kettle")
    ap.add_argument("--test-house", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--keep-off-ratio", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=12)
    args = ap.parse_args()

    slug = args.appliance.replace(" ", "_")
    sets = subsets(APPLIANCE_HOUSES[args.appliance], args.test_house)
    print("subsets:", sets)

    rows = []
    for houses in sets:
        tag = "curve" + "".join(str(h) for h in houses)
        print("\n=== training on houses %s ===" % houses, flush=True)

        run([sys.executable, "-m", "src.models.train",
             "--appliance", args.appliance,
             "--test-house", str(args.test_house),
             "--only-houses", ",".join(str(h) for h in houses),
             "--epochs", str(args.epochs),
             "--keep-off-ratio", str(args.keep_off_ratio),
             "--stride", str(args.stride),
             "--tag", tag])

        run([sys.executable, "-m", "src.evaluate.cross_house",
             "--appliance", args.appliance,
             "--test-house", str(args.test_house),
             "--checkpoint", "models/seq2point_%s_%s.pt" % (slug, tag)])

        # cross_house writes the predictions; score them here rather than
        # parsing its stdout, whose shape changes when baselines exist.
        csv = Path("data/processed/seq2point_%s_house%d.csv"
                   % (slug, args.test_house))
        d = pd.read_csv(csv, index_col=0, parse_dates=True)
        s = score(d["seq2point"], d["truth"], args.appliance)
        rows.append({"n_houses": len(houses), "houses": houses,
                     "mae": round(s["mae"], 2), "sae": round(s["sae"], 3),
                     "precision": s["precision"], "recall": s["recall"],
                     "f1": s["f1"]})
        print("  -> F1 %.3f" % s["f1"], flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2))
    print("\nwrote", RESULTS)


if __name__ == "__main__":
    main()