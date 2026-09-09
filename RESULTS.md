# Results

## Protocol

Train on UK-DALE houses 1-4, test on unseen house 5.
Per-house random splits are not used - with sliding windows they place
near-duplicate samples on both sides and inflate results.

Test window per appliance, because usage frequency differs by orders of
magnitude:
- kettle: 2014-07-01 to 07-08 (100,202 samples)
- microwave, dish washer: 2014-08-01 to 09-01 (445,802 samples)

House 5's microwave ran ~100 minutes across 4.5 months, so a single week can
contain zero events and there is nothing to score against.

Mains loaded as apparent power for all models.
seq2point predictions are clamped to >= 0 W.

## Kettle, house 5

| Model | MAE (W) | SAE | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| NILMTK CO | 214.3 | 12.80 | 0.051 | 0.976 | 0.097 | 485 | 8970 | 12 |
| NILMTK FHMM | 193.8 | 11.46 | 0.061 | 0.972 | 0.114 | 483 | 7490 | 14 |
| seq2point (keep_off_ratio 4) | 100.1 | 2.23 | 0.242 | 1.000 | 0.390 | 497 | 1553 | 0 |
| **seq2point (keep_off_ratio 20)** | **28.2** | 1.13 | **0.415** | 0.861 | **0.560** | 428 | 604 | 69 |
| seq2point dilated, RF 502 | 36.7 | 1.10 | 0.328 | 0.913 | 0.483 | 454 | 929 | 43 |

All rows at a 1000 W decision threshold. See "Decision threshold" below for
the best result (F1 0.640).

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

Note: this is two settings compared on the test house, not a tuning sweep.

## Dilation ablation

Hypothesis: the published conv stack has a receptive field of only 30 samples
(3 min at 6s), so it cannot see event duration - the discriminator that CO
lacks. Dilations (1,4,16,32,64) widen this to 502 samples for 0.3% more
parameters.

Result: dilation made cross-house F1 worse, 0.483 vs 0.560. Both models
reached the same best validation loss (0.00252) on training houses, so the
difference is in generalisation, not fit.

Hypothesis for why: with 50 minutes of context the model can condition on
house-level state (how busy the house is, time of day) that holds in houses
1-4 and does not transfer to house 5's very different load profile. The
narrow receptive field can only see local event shape, which is the part
that is universal across houses.

Single run per config, one seed. Treat as suggestive, not established.

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
The threshold was selected on house 4 and applied once to house 5, not
swept on the test house.

## Prediction clamping

Negative predictions clamped to zero at prediction time. MAE 36.9 -> 28.2 W.
State metrics unchanged - the negatives were all below the decision threshold.

SAE went 0.606 -> 1.134. The pre-clamp figure was flattered by negative
predictions cancelling part of the over-prediction; 1.134 is the honest
energy error and matters for the Phase 5 costing head.

## Failed experiment: output ReLU

Predictions can go negative, which is physically impossible. Adding nn.ReLU()
as the final layer killed training: loss frozen at train 0.03485 / val 0.03094
from epoch 1 through 10, identical to five decimals. The output went negative
early, ReLU zeroed the gradient, and no gradient reached the network - the
dead ReLU problem. Reverted; negatives are clamped at prediction time instead.

## Microwave: where the pipeline breaks

Same pipeline, same protocol, house 5 test window 2014-08-01 to 09-01
(445,802 samples, 382 positives = 0.086%).

| Model | MAE (W) | SAE | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CO | 330.4 | 6.51 | 0.006 | 0.997 | 0.011 |
| FHMM | 592.5 | 11.66 | 0.001 | 1.000 | 0.002 |
| seq2point @ 200 W | 70.1 | 0.28 | 0.012 | 0.895 | 0.024 |
| seq2point @ 500 W | 70.1 | 0.28 | 0.025 | 0.743 | 0.049 |

Threshold swept on house 4 (peak F1 0.536 at 500 W) and applied once to
house 5, as for kettle.

seq2point still beats both baselines on every metric, but 0.049 is not a
working detector.

There is an arithmetic floor here. At 0.086% positives, a 1% false-positive
rate yields 4,450 false alarms against 382 real events, capping precision
near 0.08 regardless of architecture. Reaching precision 0.5 would require a
false-positive rate below 0.09%.

## Three appliances: what transfers

Threshold swept on a training house, applied once to house 5.

| Appliance | Peak W | Positives | Train house | House 5 | Drop |
|---|---|---|---|---|---|
| Kettle | 2900 | 0.50% | 0.790 (h4) | **0.640** | -19% |
| Dish washer | 1730 | 2.45% | 0.891 (h2) | **0.456** | -49% |
| Microwave | 1500 | 0.086% | 0.536 (h4) | **0.049** | -91% |

seq2point beats both baselines on all three appliances and on every metric.

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
with zero false positives across a 1650 W range of thresholds. It also
made FHMM appear to beat seq2point (0.221 vs 0.187).

Reduced to 60 s, matching the other appliances. Under the corrected metric
seq2point wins at every threshold. No retraining was needed - only the
scoring was wrong.

### What determines transfer

Two factors, both visible above. Amplitude relative to the building's noise
floor: house 5's baseline sits at 600-1000 W and swings by thousands, so a
2900 W kettle stays visible while a 1500 W microwave does not. And event
duration: a dishwasher runs 90 minutes, a microwave 90 seconds.

For a commercial building this favours chillers, AHUs and large motors -
high power and long running, the good corner of both axes. Small plug loads
are the opposite corner and this method should not be expected to find them.

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

The 1200 s minimum is a domain constraint, not a tuned parameter: a
dishwasher cycle is longer than 20 minutes, so shorter detections are noise
regardless of what the model outputs.

## End-to-end waste detection

seq2point -> runs -> occupancy flagging. Dish washer, house 5, August 2014.
Occupancy declared as 07:00-23:00 weekdays, 08:00-24:00 weekends, local time.

| | Runs | Waste runs | Total kWh | Wasted kWh | Worst hour |
|---|---|---|---|---|---|
| Predicted | 23 | 1 | 11.95 | 0.482 | 23:00 |
| Actual | 13 | 1 | 13.43 | 1.052 | 23:00 |

The single detected waste run (2014-08-01 23:25 local) is the real one.

A run counts as waste only if it STARTED outside occupancy hours. Fraction-
outside alone flagged a cycle started at 22:52 that overran past 23:00 - a
person was awake to start it, so it is not waste. Start time is the test that
matches the business question.

Limitations: total energy is 11% low and detected waste energy 54% low,
because the model catches part of each run rather than all of it. And house 5
had exactly one out-of-hours dishwasher cycle in the month, so this validates
the plumbing rather than the detector's accuracy.

## Failed experiment: LSTM state model

Hypothesis: converting seq2point's noisy power to ON/OFF with a fixed
threshold decides each timestep in isolation - the same weakness that makes
CO fire on long plateaus. A bidirectional LSTM carries state and should
learn that a dishwasher stays on once started.

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

Every LSTM configuration converged to the same failure: recall ~0.99,
precision ~0.47. It fires on anything, which is what a class-weighted loss
rewards when positives are 2.4% of samples. Lowering pos_weight from 40.5 to
5 did not change the converged solution.

One observation not counted as a result: at epoch 2 of the pos_weight 5 run,
val F1 reached 0.805 with precision 0.997, then collapsed to 0.48 precision
by epoch 3. Checkpointing on validation loss rather than F1 meant this state
was not saved - lower BCE loss corresponded to worse F1. The two objectives
disagree here.

Conclusion: the threshold wins. seq2point's output is clean enough that
sequence modelling adds nothing, and the fixed rule is more stable than a
learned one on this task.

## Costing output

Full pipeline on house 5, August 2014. Tariff is an illustrative Delhi
commercial ToD structure: base INR 8.50/kWh, +20% 17:00-23:00, -20%
23:00-06:00. The consumption data is residential UK, so these figures
demonstrate the costing method rather than reproduce a real bill.

| | Runs | Waste runs | Total energy | Wasted | Annualised |
|---|---|---|---|---|---|
| Detected | 23 | 1 | 11.95 kWh (INR 111.99) | 0.48 kWh (INR 3.28) | INR 38.62 |
| Actual | 13 | 1 | 13.43 kWh (INR 124.88) | 1.05 kWh (INR 7.16) | INR 84.30 |

The right event is found - detected 23:25, actual 23:14 - but only 23 of its
85 minutes are captured, so the cost is understated by 54%. Total monthly
energy is within 10%, so the aggregate picture is more reliable than any
individual run boundary.

The overnight off-peak rebate applies here: this waste is billed at INR
6.80/kWh rather than 8.50, which reduces the figure. A costing head that
only inflated the number would not be useful for deciding where to spend.

The absolute amounts are small because one domestic dishwasher is a small
load. The method scales with load size, not with itself.

### Seed variance

Three seeds (0, 1, 2), same config, kettle house 5:

| Threshold | F1 mean ± std |
|---|---|
| 1000 W | 0.538 ± 0.018 |
| **1550 W** | **0.615 ± 0.004** |

The swept threshold is not only better, it is far more stable across seeds
(± 0.004 vs ± 0.018). At 1000 W the model sits in a region where random
initialisation changes the decision; at 1550 W it does not.

MAE varies more than F1 does: 28.4, 35.3, 41.9 W across the three seeds, a
47% spread. SAE ranges 1.19 to 2.11. The energy estimate is less stable than
the detection, which matters because Phase 5 costing depends on energy, not
on F1.

An earlier single run reported 0.640. That was the top of the seed range,
not the mean.