# Random Seeds, Replication, and Statistical Interpretation

## Why a seed exists

A deep-learning training result is not a single deterministic calculation. Even
with the same EDF files and hyperparameters, the final checkpoint can change
because of random weight initialization, mini-batch order, weighted sampling
with replacement, and dropout masks. Some GPU operations can also introduce
small implementation-dependent variation.

A **random seed** fixes the initial state of the random-number generators for
one reproducible trial. In this repository, `CHBMIT_TRAIN_SEED` is persisted in
each run's `hyperparameters.json` and seeds Python `random`, NumPy, PyTorch CPU,
and all available PyTorch CUDA generators. The prepared NPZ files, locked
recording split, preprocessing configuration, validation set, architecture, and
hyperparameters remain constant across the seed trials.

Thus, `seed=42`, `seed=7`, and `seed=123` are **three independent training
replicates**, not three cross-validation folds and not three different patient
cohorts. A seed controls much of the training randomness but does not by itself
guarantee bitwise-identical results across different hardware, PyTorch/CUDA
versions, or non-deterministic CUDA kernels.

## Why one lucky run is not a result

Let `m_s` be a validation metric from seed `s`. A paper should report at least
the sample mean and sample standard deviation:

```text
mean = (m_1 + ... + m_n) / n
sd   = sqrt(sum((m_s - mean)^2) / (n - 1))
```

The mean estimates expected performance under the declared training procedure;
the standard deviation estimates its sensitivity to training randomness. A
single best seed estimates neither and can silently select an unusually
favourable initialization. Reproducibility guidance explicitly asks authors to
state the number of runs, random-seed handling, and uncertainty rather than
presenting an unqualified headline value.

Three seeds are a practical **minimum confirmation gate** for this GPU-limited
research cycle. They are enough to catch an obvious seed-42-only effect and to
record a first variability estimate. They are not a magic scientific threshold,
nor are they sufficient to establish a precise confidence interval or a strong
null-hypothesis claim. Final paper evidence should expand the frozen winner and
the frozen baseline to five independent seeds when compute permits.

## Current worked example: R2 Lite, 2 s versus 5 s

The following comparison is paired by training seed. Each row uses the same
architecture and optimizer; only the raw input duration and therefore the
fully-ictal validation window population differ.

| Seed | R2 Lite 2 s accuracy | R2 Lite 5 s accuracy | 5 s minus 2 s |
|---:|---:|---:|---:|
| 42 | 91.175% | 92.830% | +1.655 pp |
| 7 | 89.815% | 92.132% | +2.317 pp |
| 123 | 89.610% | 94.280% | +4.670 pp |
| Mean +/- SD | 90.200% +/- 0.850% | **93.081% +/- 1.096%** | **+2.881% +/- 1.585 pp** |

All three directions are positive. This is robust enough to promote raw 5 s
R2 Lite from a seed-42 screen to the current accuracy candidate. It does **not**
prove a statistically significant improvement: with only three observations,
the two-sided 95% t interval around the paired delta is wide and includes zero.
Furthermore, the 2 s and 5 s validation windows are not identical, so the
delta is a context-ablation direction rather than a direct same-example test.

The correct claim at this point is: *under the locked within-case protocol, the
5-second raw-context ablation improved all three matched seed runs and raised
mean balanced validation-window accuracy to 93.081% +/- 1.096%.* It is not
permitted to claim 94.280% as the model result, 95% as achieved, clinical
generalisation, statistical superiority, or patient-independent performance.

## Research gates in this project

| Stage | Runs | Purpose | Allowed conclusion |
|---|---|---|---|
| Screen | Seed 42 | Cheap rejection of a controlled change | Candidate is promising or rejected |
| Confirmation | Seeds 42, 7, 123 | Measure initial training variance | Candidate mean +/- SD; no best-seed selection |
| Freeze | Five seeds for frozen winner and baseline | Narrow uncertainty and enable a paired comparison | Reproducible development result |
| Clinical evaluation | Frozen configuration on untouched patient-held-out test | Measure generalisation and event behaviour | Clinical generalisation only within that declared protocol |
| Hardware evaluation | One frozen checkpoint and INT16 package on KV260 | Measure PPA and functional agreement | Deployment evidence, not new AI selection |

For a final event-level metric, resampling must be by recording or patient
blocks, not by overlapping 1-second windows, because neighbouring windows are
strongly correlated. Patient-level or recording-level bootstrap confidence
intervals and per-patient tables are therefore more meaningful than a naive
window-level confidence interval.

## Reporting requirements for the manuscript

1. List every seed, its best epoch, and its validation metrics in supplementary
   material; do not hide unsuccessful replicates.
2. State that model selection uses validation loss, not the maximum sampled
   accuracy epoch.
3. Give mean +/- SD across seeds in the main table and explicitly name the
   fixed split, preprocessing, input length, class sampling, and model size.
4. Separate seed variation from split variation: repeated seeds do not validate
   new patients or new recordings.
5. Freeze the architecture, all hyperparameters, and threshold/policy before
   any untouched-test or KV260 measurement.
6. Release configuration, run manifests, checkpoint hashes, software versions,
   and all result artifacts needed to reproduce the main result.

## Sources

- [JMLR reproducibility best practice: perform multiple runs with different seeds](https://www.jmlr.org/papers/volume21/20-056/20-056.pdf)
- [AAAI reproducibility checklist: describe seed handling and number of runs](https://aaai.org/conference/aaai/aaai-23/reproducibility-checklist/)
- [Pineau et al., 2021, Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html)
- [TRIPOD+AI reporting guidance for machine-learning clinical prediction models](https://www.bmj.com/content/385/bmj-2023-078378)
