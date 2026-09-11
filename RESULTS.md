# Results

## Protocol

Train on UK-DALE houses 1-4, test on unseen house 5. A second split holding
out house 2 is reported separately below.

Per-house random splits are not used - with sliding windows they place
near-duplicate samples on both sides and inflate results.

Test window per appliance, because usage frequency differs by orders of
magnitude:
- kettle: 2014-07-01 to 07-08 (100,202 samples)
- microwave, dish washer, washer dryer: 2014-08-01 to 09-01 (445,802 samples)

House 5's microwave ran ~100 minutes across 4.5 months, so a single week can
contain zero events and there is nothing to score against.

Mains loaded as apparent power for all models. seq2point predictions are
clamped to >= 0 W.

## Method

```
UK-DALE HDF5
   -> load_ukdale.py    read the file directly (no NILMTK dependency)
   -> align.py          mains + appliance on a common 6-second grid
   -> windows.py        599-sample windows, midpoint target
   -> seq2point.py      CNN, 30.5M params: aggregate -> appliance power
   -> metrics.py        power -> ON/OFF states
   -> runs.py           states -> discrete runs, widened against raw mains
   -> waste.py          which runs started outside occupancy hours
   -> tariff.py         waste -> rupees under a time-of-day tariff
```

Splits are per appliance, because the houses do not record the same
appliances or use the same labels. House 5 calls its fridge a `fridge
freezer` and has no `washing machine`, only a `washer dryer`. See
`appliance_availability()` in `src/data/load_ukdale.py`.

The normaliser is fitted on training houses only and stored in the
checkpoint. Decision thresholds are swept on a held-out period of a training
house and applied once to the test house, never swept on test.

## Kettle, house 5

| Model | MAE (W) | SAE | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| NILMTK CO | 214.3 | 12.80 | 0.051 | 0.976 | 0.097 | 485 | 8970 | 12 |
| NILMTK FHMM | 193.8 | 11.46 | 0.061 | 0.972 | 0.114 | 483 | 7490 | 14 |
| seq2point (keep_off_ratio 4) | 100.1 | 2.23 | 0.242 | 1.000 | 0.390 | 497 | 1553 | 0 |
| **seq2point (keep_off_ratio 20)** | **28.2** | 1.13 | **0.415** | 0.861 | **0.560** | 428 | 604 | 69 |
| seq2point dilated, RF 502 | 36.7 | 1.10 | 0.328 | 0.913 | 0.483 | 454 | 929 | 43 |

All rows at a 1000 W decision threshold. The headline result (0.615) uses the
swept 1550 W threshold - see below.

NILMTK 0.4.3 (git 303d45b), Python 3.7 under Rosetta.

## Why the baselines fail

CO's false positives sit at a single power level (~2200 W, its lower k-means
cluster) and last 15-20 minutes. Real kettle events last 2-4 minutes. CO
decides each timestep independently from power level alone and has no
representation of event duration.

## Class balancing

~0.9% of training windows have the kettle on. Under plain MSE, predicting
zero everywhere is near-optimal: low MAE, zero F1. keep_off_ratio keeps all
positive windows and subsamples negatives.

Two settings were compared on house 5. At ratio 4 the model saw 20% positives
against a test reality of 0.5%, giving perfect recall and poor precision
(F1 0.390). At ratio 20 it saw ~5%, and precision nearly doubled (F1 0.560).

Note: two settings compared on the test house, not a tuning sweep.

## Decision threshold

Predictions are a continuous regression, so turning them into ON/OFF states
needs a decision threshold. This does not have to equal the on_power used to
interpret ground truth.

Swept 250-3000 W on held-out house 4 data (a training house, unseen period).
Peak F1 0.790 at 1550 W, vs 0.724 at 1000 W. The peak is a plateau -
1400-1650 W are all within 0.008 - so the choice is not fitted to noise.

Applying 1550 W once to house 5:

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 1000 W | 0.415 | 0.861 | 0.560 | 428 | 604 | 69 |
| **1550 W** | **0.555** | 0.757 | **0.640** | 376 | 302 | 121 |

Both rows are the same model with negative predictions clamped to zero.
The threshold was selected on house 4 and applied once to house 5.

### Seed variance

Three seeds (0, 1, 2), same config, kettle house 5:

| Threshold | F1 mean +/- std |
|---|---|
| 1000 W | 0.538 +/- 0.018 |
| **1550 W** | **0.615 +/- 0.004** |

The swept threshold is not only better, it is far more stable across seeds
(+/- 0.004 vs +/- 0.018). At 1000 W the model sits in a region where random
initialisation changes the decision; at 1550 W it does not.

MAE varies more than F1 does: 28.4, 35.3, 41.9 W across the three seeds, a
47% spread. SAE ranges 1.19 to 2.11. The energy estimate is less stable than
the detection, which matters because costing depends on energy, not on F1.

An earlier single run reported 0.640. That was the top of the seed range,
not the mean.

## Prediction clamping

Negative predictions clamped to zero at prediction time. MAE 36.9 -> 28.2 W.
State metrics unchanged - the negatives were all below the decision threshold.

SAE went 0.606 -> 1.134. The pre-clamp figure was flattered by negative
predictions cancelling part of the over-prediction; 1.134 is the honest
energy error.

## Dilation ablation

Hypothesis: the published conv stack has a receptive field of only 30 samples
(3 min at 6s), so it cannot see event duration - the discriminator that CO
lacks. Dilations (1,4,16,32,64) widen this to 502 samples for 0.3% more
parameters.

Result: dilation made cross-house F1 worse, 0.483 vs 0.560. Both models
reached the same best validation loss (0.00252) on training houses, so the
difference is in generalisation, not fit.

Hypothesis for why: with 50 minutes of context the model can condition on
house-level state that holds in houses 1-4 and does not transfer to house 5.
The narrow receptive field can only see local event shape, which is the part
that is universal across houses.

Single run per config, one seed. Suggestive, not established.

## Failed experiment: output ReLU

Predictions can go negative, which is physically impossible. Adding nn.ReLU()
as the final layer killed training: loss frozen at train 0.03485 / val 0.03094
from epoch 1 through 10, identical to five decimals. The output went negative
early, ReLU zeroed the gradient, and no gradient reached the network - the
dead ReLU problem. Reverted; negatives are clamped at prediction time instead.

## Failed experiment: LSTM state model

Hypothesis: converting seq2point's noisy power to ON/OFF with a fixed
threshold decides each timestep in isolation - the same weakness that makes
CO fire on long plateaus. A bidirectional LSTM carries state and should learn
that a dishwasher stays on once started.

Setup: 2-layer bidirectional LSTM, 133,761 parameters, 512-sample sequences
(~51 min). Trained on seq2point's predictions over held-out periods of house
1, validated on house 2 - so it learns to clean up this model's actual error
pattern, not idealised power. BCEWithLogitsLoss with pos_weight.

Result, sample-level F1 on house 2:

| Method | F1 | Precision | Recall |
|---|---|---|---|
| **Threshold @ 100 W** | **0.868** | 0.891 | 0.846 |
| Threshold @ 50 W | 0.791 | 0.696 | 0.916 |
| LSTM, pos_weight 40.5 (inverse freq) | 0.643 | 0.477 | 0.987 |
| LSTM, pos_weight 10 | 0.643 | 0.477 | 0.987 |
| LSTM, pos_weight 5 | 0.641 | 0.474 | 0.987 |

Every configuration converged to the same failure: recall ~0.99, precision
~0.47. It fires on anything, which is what a class-weighted loss rewards when
positives are 2.4% of samples. Lowering pos_weight from 40.5 to 5 did not
change the converged solution.

One observation not counted as a result: at epoch 2 of the pos_weight 5 run,
val F1 reached 0.805 with precision 0.997, then collapsed to 0.48 precision
by epoch 3. Checkpointing on validation loss rather than F1 meant this state
was not saved - lower BCE loss corresponded to worse F1. The two objectives
disagree here.

Conclusion: the threshold wins. seq2point's output is clean enough that
sequence modelling adds nothing, and the fixed rule is more stable.

## Four appliances

Threshold swept on a training house, applied once to the test house.

| Appliance | Peak W | Train house F1 | House 5 F1 | Drop |
|---|---|---|---|---|
| Kettle | 3996 | 0.880 (h1) | 0.615 | -30% |
| Dish washer | 1730 | 0.734 (h1) | 0.456 | -38% |
| Washer dryer | 2487 | 0.960 (h1) | 0.233 | -76% |
| Microwave | 1562 | 0.605 (h1) | 0.049 | -92% |

seq2point beats both baselines on every appliance and every metric.

### Dish washer, house 5, 2014-08-01 to 09-01

| Model | MAE (W) | SAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CO | 313.6 | 17.32 | 0.025 | 1.000 | 0.048 |
| FHMM | 227.0 | 12.45 | 0.132 | 0.637 | 0.218 |
| seq2point @ 10 W | 20.0 | 0.63 | 0.161 | 0.903 | 0.274 |
| **seq2point @ 100 W** | 20.0 | 0.63 | 0.385 | 0.559 | **0.456** |

### Metric bug found and fixed

The dish washer config originally used min_on_s = min_off_s = 1800, taken
from NILM convention. Applied symmetrically to truth and predictions, this
bridged the quiet phases mid-cycle in both, inflating truth by 31% and
predictions by 43%, and producing a degenerate precision of exactly 1.000
with zero false positives across a 1650 W range of thresholds. It also made
FHMM appear to beat seq2point (0.221 vs 0.187).

Reduced to 60 s, matching the other appliances. Under the corrected metric
seq2point wins at every threshold. No retraining was needed - only the
scoring was wrong.

## Two test houses

Same models, same protocol, threshold swept on a held-out period of house 1
and applied once to each test house.

| Appliance | House 5 F1 | House 2 F1 | Gap |
|---|---|---|---|
| Kettle | 0.615 | **0.904** | +0.29 |
| Dish washer | 0.456 | **0.747** | +0.29 |
| Microwave | 0.049 | **0.478** | +0.43 |

The appliance ordering holds in both houses, but the level is set by the test
building, and that effect is larger than the appliance effect.

House 5 has 26 meters and 22 appliances including an electric oven, electric
stove, server computer and NAS. Its aggregate baseline sits at 600-1000 W and
swings by thousands. House 2 has 20 meters and 17 appliances and is quieter.

The quieter building helps the weakest appliance most: microwave gains 0.43,
the high-power kettle only 0.29.

### Correction to an earlier conclusion

An earlier draft attributed the microwave failure to power and duration alone,
and argued an arithmetic floor at 0.086% positives capped precision near 0.08
regardless of architecture. House 2's microwave reaches precision 0.454. That
floor was a property of house 5's event rate in that specific window, not of
microwaves.

### Thresholds transfer even when scores do not

Swept on house 1, the winning thresholds were 1500 W (kettle), 100 W (dish
washer) and 950 W (microwave). House 5's sweep gave 1550 W and 100 W for the
first two. The decision boundary is a property of the model; the achievable
score is a property of the test house.

## What predicts transfer

Seven measurements: four appliances on house 5, three on house 2, all at
swept thresholds.

| Appliance | Test house | Peak W | Noise W | SNR | Duration | Density | Train houses | F1 |
|---|---|---|---|---|---|---|---|---|
| kettle | 5 | 3996 | 648 | 6.16 | 1.8 min | 0.41% | 4 | 0.615 |
| dish washer | 5 | 1730 | 537 | 3.22 | 16.3 min | 1.01% | 2 | 0.456 |
| microwave | 5 | 1562 | 688 | 2.27 | 0.9 min | 0.08% | 3 | 0.049 |
| washer dryer | 5 | 2487 | 550 | 4.52 | 0.2 min | 1.17% | 1 | 0.233 |
| kettle | 2 | 3993 | 422 | 9.45 | 2.9 min | 0.78% | 4 | 0.904 |
| dish washer | 2 | 3955 | 337 | 11.75 | 13.5 min | 1.99% | 2 | 0.747 |
| microwave | 2 | 2500 | 520 | 4.80 | 1.2 min | 0.52% | 3 | 0.478 |

SNR is appliance peak power divided by the standard deviation of the
aggregate while that appliance is off - how loud the load is relative to the
building it sits in.

Correlation with F1:

| Variable | r |
|---|---|
| SNR x train houses | 0.893 |
| log(SNR) | 0.885 |
| SNR | 0.841 |
| train houses | 0.432 |
| event density | 0.423 |
| run duration | 0.307 |

SNR dominates. Duration and event density, which an earlier draft treated as
the explanation, rank last.

Two rows show why one variable is not enough. Washer dryer on house 5 has
SNR 4.52 and F1 0.233; kettle on house 5 has SNR 6.16 and F1 0.615. Similar
SNR, very different scores - the difference is 1 training house against 4.
Dish washer on house 2 has the highest SNR in the table (11.75) but scores
below kettle at 9.45, so SNR does not order the top end either.

The combined term and log(SNR) score 0.893 and 0.885 - indistinguishable with
seven points. This data cannot separate "SNR with diminishing returns" from
"SNR and training-house count together".

### Measurement fragility

An earlier version used the 99.9th percentile for peak power. At house 5's
microwave density of 0.08%, that percentile sits below the real peaks and
reported 59 W instead of 1562 W. Fixing that one row moved the SNR
correlation from 0.929 to 0.841.

Seven points, four candidate variables. No model is fitted here and none
should be. What the data supports is a ranking and an explanation of the
outliers.

## How many training buildings

Kettle, test house 5, one nested subset per size (ordered by data volume so
each step is a superset of the last).

| Training houses | F1 | MAE | SAE |
|---|---|---|---|
| 1 | 0.451 | 36.3 | 1.38 |
| **2** | **0.560** | **25.2** | **0.87** |
| 3 | 0.547 | 27.5 | 1.10 |
| 4 | 0.548 | 36.5 | 1.78 |

One house to two gains 0.109. Three and four gain nothing - both sit inside
the +/- 0.018 seed noise band. MAE and SAE are actually worst at four houses.

Variety of buildings matters; volume of data does not. A second building
forces the model to learn what is common to kettles rather than what is
specific to house 1's kettle.

This also explains the washer dryer result: it trains on house 1 alone, hits
F1 0.960 there, and collapses to 0.233 on house 5 - the largest transfer gap
in the project despite favourable power, duration and event density.

Single seed per point, and houses are added in a fixed order rather than
averaged over all 15 subsets.

## Run extension

Detected runs come out short: the model's output dips below threshold at the
edges of a real run while holding above it in the middle. `extend_runs` walks
each detected run outward while the raw aggregate stays one appliance-power
above a local baseline (20th percentile over a 3-hour window, so the run
itself does not contaminate the baseline).

Dish washer, house 5, August 2014:

| | Total energy | vs actual 13.43 kWh |
|---|---|---|
| No extension | 11.95 kWh | -11.0% |
| **margin 1.0** | **13.95 kWh** | **+3.9%** |
| margin 1.5 | 12.79 kWh | -4.8% |
| margin 0.5 | 19.35 kWh | +44.1% |

margin 1.0 is the principled value - extend while the aggregate holds a full
appliance-power above baseline - not a tuned one. The other rows are a
sensitivity check.

Run boundaries are still wrong: mean detected duration is ~1980 s against a
true ~5300 s for most cycles. The energy total is close because the model
finds a shorter, higher-power slice.

## Run-level accuracy

Sample-level F1 understates usable performance. Scattered false-positive
samples collapse into few runs once a minimum duration is required.

Dish washer, house 5, August 2014. 13 actual runs.

| min_duration | Detected | TP | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0 s | 193 | 13 | 180 | 0.067 | 1.000 | 0.126 |
| 600 s | 62 | 13 | 49 | 0.210 | 1.000 | 0.347 |
| **1200 s** | 23 | 12 | 11 | **0.522** | 0.923 | **0.667** |

Sample-level F1 for the same predictions is 0.456.

The 1200 s minimum is a domain constraint, not a tuned parameter: a dishwasher
cycle is longer than 20 minutes, so shorter detections are noise regardless of
what the model outputs.

## End-to-end costing

seq2point -> runs -> occupancy -> tariff. Dish washer, house 5, August 2014.
Occupancy declared as 07:00-23:00 weekdays, 08:00-24:00 weekends, local time,
with 15 minutes grace either side. Tariff is an illustrative Delhi commercial
ToD structure: base INR 8.50/kWh, +20% 17:00-23:00, -20% 23:00-06:00.

| | Runs | Waste runs | Total energy | Wasted | Annualised |
|---|---|---|---|---|---|
| Detected | 23 | 1 | 13.95 kWh (INR 131.92) | 0.48 kWh (INR 3.28) | INR 38.62 |
| Actual | 13 | 0 | 13.43 kWh (INR 124.88) | 0.00 kWh | - |

A run counts as waste only if it STARTED outside occupancy hours. Fraction-
outside alone flagged a cycle started at 22:52 that overran past 23:00 - a
person was awake to start it, so it is not waste.

The overnight off-peak rebate applies to this finding: it is billed at INR
6.80/kWh rather than 8.50. A costing head that only inflated the number would
not be useful for deciding where to spend.

The absolute amounts are small because one domestic dishwasher is a small
load. The method scales with load size.

## Scope

There is no public commercial NILM dataset with paired aggregate and
per-circuit ground truth - without it there is nothing to train against or
score. So this validates the method on residential UK-DALE and states the
transfer gap rather than glossing over it.

The SNR result gives that gap a measurable form. High-power, long-running
loads - chillers, air handling units, large motors - sit in the favourable
region. Small plug loads do not, and this method should not be expected to
find them.

## Known limitations

- The single waste finding above is a **false positive**. Ground truth for
  that run starts at 23:14:48 BST - twelve seconds inside the occupancy
  window with grace applied, so it is not waste. The model detects it at
  23:25, eleven minutes late, which puts it outside. House 5's August
  contains no genuine out-of-hours dishwasher runs under this schedule, so
  the waste detector has nothing in this window to validate against.
- Costing understates individual runs by ~54% for the same reason: the model
  catches part of each run. Monthly totals are within 4%.
- Microwave does not transfer to house 5 (F1 0.049).
- The dilation ablation, the LSTM state model and the house curve are single
  runs. Only the kettle headline result has three seeds.
- The transfer characterisation rests on seven measurements and four
  candidate variables. It is a ranking, not a model.