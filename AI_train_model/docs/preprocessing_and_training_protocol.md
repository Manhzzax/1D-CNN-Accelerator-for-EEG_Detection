# CHB-MIT Preprocessing And Training Protocol

## Locked Baseline

The experiment uses 686 locally verified EDF recordings from CHB-MIT v1.0.0. The database contains pediatric long-term scalp EEG recordings and official seizure annotations. See the [PhysioNet dataset record](https://physionet.org/content/chbmit/1.0.0/).

1. Resolve every recording to the fixed 17-channel bipolar montage.
2. Lock the case-wise chronological recording split before extracting any window.
3. Read and filter the complete recording, then create 1-second windows at 256 Hz with 1-second stride.
4. Mark an ictal window only when it is fully contained in an official seizure interval. Exclude normal windows inside a 30-second guard around every seizure.
5. Fit one mean and standard deviation per channel from train windows only. Transform all splits with `x[c] = (x[c] - mean_train[c]) / max(std_train[c], eps)`.
6. Use validation for model, threshold, and temporal-policy selection. Use the continuous test recordings only for the final report.

The current filter is a fourth-order zero-phase 0.5-45 Hz band-pass over each complete offline recording. This avoids per-window filter-edge artifacts, consistent with MNE guidance to filter continuous data before epoching. The current 60 Hz notch is redundant after a 45 Hz low-pass, so it is retained only for reproducibility and must be removed or redesigned in the next preprocessing ablation. MNE documents notch filtering and the benefit of operating on continuous data in its [filtering guide](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html).

## Normalization Decision

Train-fitted, channel-wise z-score is the primary normalization. It is leakage-safe, preserves the fixed channel order, and is compatible with a fixed-point FPGA implementation because the affine scale can be folded into the first convolution.

Do not fit a separate scaler on validation or test. Do not fit one scaler over the whole CHB-MIT corpus. Both choices leak held-out distribution statistics.

Two ablations are justified after the mixed hard-negative run:

| Variant | Fit scope | Rationale | Deployment caveat |
|---|---|---|---|
| Train channel-wise z-score | Train windows only | Primary baseline; current implementation | None beyond storing 17 means and scales |
| Per-recording z-score | Each recording independently | Tests robustness to patient amplitude differences | A full-recording statistic is offline; online use requires a causal running estimate |
| Train robust scaling | Train median and IQR per channel | Limits influence of artifacts and extreme hard negatives | Requires percentile constants in hardware |

Per-file z-scoring has also been studied specifically with CHB-MIT, but it is a separate protocol and must not be mixed with the primary result without a dedicated ablation. See [SeizyML](https://pmc.ncbi.nlm.nih.gov/articles/PMC11876212/).

## Early Stopping

The trainer monitors validation cross-entropy loss, saves the lowest-loss checkpoint, and stops after six consecutive epochs without at least `0.001` loss improvement, but never before epoch eight. The learning-rate scheduler remains active before the stop decision.

This is a compute-saving regularizer, not a substitute for event-level selection. Final model comparison remains based on validation event sensitivity, false alarms per hour, and delay. The basis for validation-based stopping is [Prechelt, 1998](https://pubmed.ncbi.nlm.nih.gov/12662814/), which evaluates automatic cross-validation stopping criteria and their time-generalization trade-off.
