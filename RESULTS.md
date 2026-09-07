# Results

## Protocol

Train on UK-DALE houses 1-4, test on unseen house 5.
Test window 2014-07-01 to 2014-07-08, 100,202 samples at 6s.
Thresholds for state metrics: kettle on_power 1000 W, min_on 12 s.
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