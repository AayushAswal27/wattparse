"""Score a trained seq2point model on the held-out house.

    python -m src.evaluate.cross_house --appliance kettle

Loads the checkpoint's normaliser rather than refitting - refitting on the
test house would leak its distribution into the numbers. No balancing here
either: balancing is a training trick, and the test set has to be the real
distribution or the F1 is not comparable to the baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.align import align
from src.data.windows import WINDOW, Normaliser, make_windows
from src.evaluate.metrics import compare, score
from src.models.seq2point import Seq2Point

H5 = "data/raw/ukdale.h5"
TEST_HOUSE = 5
TEST_WINDOW = ("2014-07-01", "2014-07-08")   # same week as the baselines


def predict(model, X, norm, device, batch_size=512):
    """Run the model over all windows and return watts."""
    Xn = norm.transform(X)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.from_numpy(Xn[i:i + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return norm.inverse_target(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default="kettle")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    appliance = args.appliance
    slug = appliance.replace(" ", "_")
    ckpt_path = Path(args.checkpoint or ("models/seq2point_%s.pt" % slug))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = Seq2Point(dilations=tuple(ckpt["dilations"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    norm = Normaliser.from_dict(ckpt["normaliser"])
    print("loaded %s (trained on %s)" % (ckpt_path, ckpt["appliance"]))
    print("normaliser:", {k: round(v, 1) for k, v in norm.to_dict().items()})

    df = align(H5, TEST_HOUSE, appliance,
               start=TEST_WINDOW[0], end=TEST_WINDOW[1])
    X, y = make_windows(df, appliance, stride=1)
    print("house %d: %d windows" % (TEST_HOUSE, len(X)))

    # Window i predicts the sample at i + WINDOW//2, so predictions land on
    # that slice of the original index - not on df.index[:len(pred)].
    half = WINDOW // 2
    idx = df.index[half:half + len(y)]

    pred = pd.Series(predict(model, X, norm, device), index=idx, name="seq2point")
    truth = pd.Series(y, index=idx, name="truth")

    print("\nseq2point on house %d, %s to %s"
          % (TEST_HOUSE, TEST_WINDOW[0], TEST_WINDOW[1]))
    print(pd.DataFrame([score(pred, truth, appliance)]).round(3).to_string(index=False))

    # Baselines cover the same week but the full index, so join on ours.
    base_path = Path("data/processed/baseline_%s.csv" % slug)
    if base_path.exists():
        base = pd.read_csv(base_path, index_col=0, parse_dates=True)
        base.index = base.index.tz_convert(idx.tz)
        base = base.reindex(idx)
        print("\ncomparison on the same %d samples:" % len(idx))
        print(compare({"CO": base["CO"],
                       "FHMM": base["FHMM"],
                       "seq2point": pred},
                      truth, appliance).round(3).to_string())
    else:
        print("\nno baseline csv at %s - run src/models/baselines.py first"
              % base_path)

    out = Path("data/processed/seq2point_%s_house%d.csv" % (slug, TEST_HOUSE))
    pd.DataFrame({"truth": truth, "seq2point": pred}).to_csv(out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()