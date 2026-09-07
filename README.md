# WattParse

Disaggregating a building's single electricity meter into per-appliance
traces, then flagging equipment running outside occupancy hours and pricing
the waste.

## The problem

Commercial buildings waste 15-30% of energy on equipment running outside
occupancy hours. Sub-metering every circuit is expensive and needs an
electrician per panel, so the only signal a facility manager has is the main
meter. They know the bill is too high; they cannot tell you what is causing
it.

## What this validates, and what it does not

There is no public commercial NILM dataset with paired aggregate and
per-circuit ground truth. Without ground truth there is nothing to train
against and nothing to score. So this project validates the **method** on
UK-DALE, which is residential, and states explicitly what would and would not
carry over.

This is not a commercial deployment. It is a method validated where ground
truth exists, with the transfer gap named rather than glossed over.

**Transfers:** the seq2point framing (many samples in, one point out), the
cross-house validation protocol, the class-balancing treatment of a rare
positive class, and the finding that the decision threshold should be
selected separately from the ground-truth threshold.

**Does not transfer:** the learned kettle signature; the 599-sample window,
sized for events lasting minutes rather than hours; single-phase assumptions;
and any tuning specific to a 20-appliance building rather than a
50-plus-load one.

**Directly relevant evidence.** Widening the receptive field from 30 to 502
samples improved nothing and made cross-house F1 worse (0.560 -> 0.483),
while reaching identical validation loss on the training houses. Wider
context lets the model condition on building-level state that does not
generalise. For transfer to a different building type, that is the finding
that matters most - and it argues for narrow, local, shape-based features
over long-range context.

## Approach

| Stage | Model | Job |
|---|---|---|
| Disaggregator | seq2point CNN | 599-sample aggregate window -> appliance power at midpoint |
| State model | threshold + duration rules | power regression -> ON/OFF states and run durations |
| Waste detector | occupancy rule, LSTM-AE as a stretch | runs outside occupancy hours |
| Costing | deterministic | anomalous runs + time-of-day tariff -> money wasted |

## Validation protocol

Cross-house: train on UK-DALE houses 1, 2, 3, 4 - test on unseen house 5.
Per-house random splits are not used; with sliding windows they place
near-duplicate samples on both sides of the split and inflate results.

Splits are per appliance, because the houses do not record the same
appliances and do not use the same labels for them. See
`appliance_availability()` in `src/data/load_ukdale.py`.

## Results

Kettle, house 5, F1 0.640 against NILMTK CO 0.097 and FHMM 0.114.
Full numbers, ablations and one failed experiment in [RESULTS.md](RESULTS.md).

## Data

UK-DALE 2017 HDF5 (~5.5 GB) from the CEDA archive. Unzip to
`data/raw/ukdale.h5`.

The file was written in 2017 and stores its pandas type attributes as bytes,
which breaks every read under modern pandas. `src/data/load_ukdale.py` reads
the PyTables nodes directly and has no NILMTK dependency.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

NILMTK is deliberately not in `requirements.txt`. It needs Python 3.7 and
pandas 0.25 and will break a modern PyTorch install. It is used only for the
Phase 2 baselines, in a separate conda environment (on Apple Silicon it
installs only under Rosetta, via `CONDA_SUBDIR=osx-64`).

## Quickstart

```bash
python -m src.data.load_ukdale data/raw/ukdale.h5    # audit the file
python -m src.models.train --appliance kettle --epochs 10 --keep-off-ratio 20
python -m src.evaluate.cross_house --appliance kettle
```

## Status

Phase 3 complete. Disaggregator beats both NILMTK baselines on the
cross-house protocol. Next: state model and waste detection.