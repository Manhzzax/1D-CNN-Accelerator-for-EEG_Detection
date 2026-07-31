# Academic Validity Audit And Benchmark Plan

**Scope.** This audit evaluates the current EpiSepNet-5K evidence, not the
quality of the implementation alone. It distinguishes a useful engineering
reference from a result that can support a Q1 clinical-detection claim.

## Verdict

The current result is a valid **internal, balanced-window and offline
screening reference**:

- EpiSepNet-5K FP32: 5,013 trainable parameters; 90.0718% validation accuracy;
  90.7645% validation seizure sensitivity; AUROC 96.5802%; 2 s raw windows.
- The selected validation alarm rule detects 23/29 events (79.31%), with
  0.4671 false alarms/hour and a 17 s median detection delay.
- INT16 emulation retains 90.0462% accuracy and 90.7645% sensitivity; the
  exported tensor package is 10,030 B.

It is **not yet publishable as a clinical or real-time superiority claim**.
The necessary corrections are protocol-related, not a matter of finding a
slightly higher accuracy number.

## Findings, Ordered By Severity

### Critical: the historical test split was used during research decisions

Runs 01--08 report test metrics and their diagnostics influenced subsequent
hard-negative, temporal, and architecture choices. Therefore that test split
is no longer an untouched holdout. The current Run 21 validation-only
selection is good practice, but it does not restore the independence of the
historical test set.

**Publication consequence:** never present those test values as a final test
result or compare them with paper leaders. Preserve them as exploratory
ablations only.

**Required remedy:** freeze EpiSepNet-5K, pre-register the next protocol and
create a fresh patient-held-out final cohort. Use a nested procedure: all
architecture, normalization constants, threshold and alarm policy are selected
inside development patients; an outer held-out patient/case is scored once.
Report repeated-seed confidence intervals. Ali et al. explicitly identify
continuous class imbalance, subject variability, and event-level detection as
the three often-missed requirements for CHB-MIT evaluation
([Ali et al., 2024](https://doi.org/10.1098/rsos.230601)).

### Critical: the current split is within-case, not patient-independent

The locked chronological split separates recordings in time, but a patient's
EEG appears in train, validation and (historically) test. The current number
therefore measures adaptation to known patients, not generalization to an
unseen patient. It is stronger than a random-window split because entire
recordings are separated, but it must not be described as cross-patient or
clinical generalization.

This distinction materially changes the expected accuracy: Ghosh et al.'s
patient-exclusive 1 s benchmark reports 90.6% for kNN and 90.5% for Random
Forest, whereas Ali et al.'s continuous cross-subject event protocol reports
72.63% (5-fold) and 75.34% (leave-one-out) event sensitivity. These results
show why high within-patient/window scores do not remove the need for a
patient-independent experiment.

### Critical: preprocessing is offline non-causal

`src/chbmit_preparation.py` applies `sosfiltfilt` and `filtfilt` to an entire
recording. These are zero-phase filters and use samples after the current
window. The alarm timestamp is correctly `window_end_causal`, but the signal
fed to the classifier is not causal. The current result is consequently an
**offline continuous evaluation**, not end-to-end real-time inference.

**Required remedy:** replace the preprocessing front end for the deployment
experiment with stateful causal IIR/FIR filtering, carry filter state across
windows, recalibrate threshold/policy only on development data, and report the
performance difference. The FPGA export already contains the same warning.

### Major: repeated use of validation creates selection optimism

Run 21 is the best among many architecture, normalization, window-length,
mining and temporal-policy choices. All those choices repeatedly interrogated
the same 29 validation events. The reported value is a model-selection score,
not an unbiased estimate. A single seed also does not quantify optimization
variance.

**Required remedy:** use at least five fixed seeds for the frozen model and
non-parametric patient/case bootstrap confidence intervals for event
sensitivity, FAR/h and delay. In the final outer-fold protocol, tune only
inside each development fold.

### Major: the 90.07% accuracy is balanced-window accuracy

The Run 21 validation NPZ contains an explicit 1:1 ictal:interictal sample.
Thus its 90.07% accuracy is informative for the sampled binary classifier, and
equals balanced accuracy here, but it is **not** continuous-monitoring
accuracy. It is inappropriate to compare it numerically with papers using
natural prevalence, other class ratios, subject-specific folds or different
segment lengths.

The current documentation must avoid saying that a non-seizure-only classifier
would obtain 90.9% on the Run 21 validation result; that statement applies to
the historical 1:10 sampled test set, not the balanced validation set. On the
balanced validation set, the trivial-class baseline is 50% accuracy.

### Major: event definition and confidence reporting need formalization

The implementation uses an event as detected if a causal alarm overlaps an
annotated ictal interval. This is a valid operational definition, but a paper
must explicitly specify: annotation source, allowable overlap, window-end time
stamp, refractory duration, post-processing policy, treatment of multiple
alarms in one seizure, total interictal hours, and micro versus macro
aggregation. It must include per-case results and uncertainty, because a
micro-average can conceal failures concentrated in a few recordings.

The continuous detector work of Chung et al. demonstrates why segment and
event metrics must be reported separately: it gives both a segment evaluation
and a continuous held-out-EDF evaluation, with smoothing/threshold/consecutive
segment post-processing ([Chung et al., 2024](https://doi.org/10.3389/fneur.2024.1389731)).

### Major: data-version and annotation-count discrepancy must be disclosed

The repository audit parsed 198 seizure intervals from the downloaded 686 EDF
files and passes its consistency checks. The current PhysioNet landing page and
several papers describe 182 annotated seizures, while papers sometimes exclude
`chb24` or three `chb12` recordings. For example, Chung et al. use 23 cases,
exclude `chb24`, and then exclude 13 seizures from three chb12 files, yielding
169 seizures. This is not automatically an error in our parser; it is a
dataset-definition discrepancy that makes unqualified claims such as "all
CHB-MIT seizures" unsafe.

**Required remedy:** add a versioned `cohort_definition.csv` to the paper
artifact with every EDF, chosen label source, interval, inclusion/exclusion
reason, and SHA-256 manifest. State precisely: "686 EDF files; 198
summary/EDF intervals parsed by our audit" rather than implying equivalence to
the 182-event cohort of another paper. Cite the official dataset separately
([PhysioNet CHB-MIT v1.0.0](https://doi.org/10.13026/C2K01R)).

### Major: hardware evidence is not FPGA evidence yet

The 5,013 parameter count and INT16 emulation are strong design facts. The
28,130 B PyTorch checkpoint is not a deployment-memory metric because it
includes serialization metadata and buffers. The reproducible primary weight
memory is 20,052 B (about 19.6 KiB) in FP32; the selected INT16 tensor package
is 10,030 B. Neither demonstrates KV260 latency, BRAM, DSP, LUT, clock rate or
energy.

**Required remedy:** call the current result *INT16 emulation*, not FPGA
quantization/deployment. Publish a separate synthesis table after HLS/RTL
implementation and bit-accurate test-vector verification.

### Moderate: the internal clinical gate is a design target, not a standard

`event sensitivity >= 90%`, `FAR/h <= 0.5`, and `median delay <= 10 s` are
reasonable engineering targets but no universal clinical acceptance threshold.
They must be called a pre-specified internal operating target. Do not phrase
failure to reach it as clinical failure, nor reaching it as clinical approval.

### Moderate: no external validation and no patient-specific adaptation study

CHB-MIT is paediatric scalp EEG from one source. A Q1 paper should explicitly
limit the claim to this cohort and, if feasible, add an external cohort or a
patient-adaptation experiment. This is especially important because patient
specificity changes outcomes substantially.

## What Is Defensible Today

Use these exact claims:

1. *EpiSepNet-5K is a compact raw-EEG separable 1D-CNN reference with 5,013
   trainable parameters.*
2. *On the locked, balanced within-case chronological validation protocol, it
   achieves 90.07% window accuracy and 90.76% ictal-window sensitivity.*
3. *With the validation-selected causal alarm policy, the offline continuous
   evaluation detects 79.31% of validation events at 0.467 false alarms/hour
   and 17 s median delay.*
4. *INT16 tensor emulation preserves the validation classifier result while
   reducing exported tensor storage to 10,030 B.*

Do **not** say current best, clinical-grade, real-time, patient-independent,
or superior to the literature.

## Recommended Benchmark Package

One table cannot defend both classifier quality and clinical usefulness. The
paper should contain three non-overlapping benchmark artifacts.

| Artifact | Main question | Required columns | What may be ranked |
|---|---|---|---|
| **Table 1: protocol-matched classifier benchmark** | Is the classifier accurate under an unseen-patient protocol? | cohort, subject split, channels, window/stride, prevalence, accuracy, balanced accuracy, AUROC, AUPRC, F1, sensitivity, specificity, parameters, MACs | Only rows with the same task and outer patient-held-out protocol |
| **Table 2: continuous clinical operating point** | Does the system detect events with an acceptable false-alarm/delay trade-off? | event definition, patients/events, event SEN with CI, FAR/h with CI, delay median/IQR, refractory, threshold/policy selected only on development data | Pareto frontier only; do not rank by accuracy |
| **Table 3: deployment benchmark** | What does the accuracy-cost trade-off cost on KV260? | precision, parameter bytes, activation bytes, MACs, FPGA clock, latency/window, throughput, LUT, FF, DSP, BRAM, power/energy, bit-exact agreement | Same board, clock and implementation flow only |

The existing CSV `chbmit_detection_accuracy_efficiency_context.csv` is a
**landscape/context table**, not Table 1. It contains transparent protocol tags
and external values with a known parameter count where possible. Its purpose is
to show that 90% lies in the reported CHB-MIT window-classification range while
EpiSepNet-5K uses far fewer parameters than several published compact/deep
models. It does not prove a causal accuracy-versus-size frontier because the
papers use different populations and splits.

### How to make the 90% point academically meaningful

The correct narrative is not "90% is acceptable because our network is
small." The evidence supports a narrower and stronger statement:

> In the published CHB-MIT literature, reported window accuracies span roughly
> 90% to 99% under heterogeneous protocols. EpiSepNet-5K reaches 90.07% with
> 5,013 parameters. It is smaller than published compact deep models with
> declared counts, for example LMPSeizNet (18,024), CAD (15,059) and a 1D-CNN
> comparator (105,538), but its score is not directly rankable against them
> until evaluated under one common protocol.

Ghosh et al. are particularly useful context: their strictly
patient-independent 1 s detector reports 90.5--90.6% accuracy. It establishes
that a value near 90% can be scientifically meaningful under a stronger
generalization protocol. It does **not** validate the present within-case
result; it defines the right next experiment.

## Evidence Used For The Expanded Landscape

The source table contains three evidence levels:

- **Direct-primary:** the paper reports its own result and protocol.
- **Direct-comparison-table:** a peer-reviewed paper's table reports cited
  methods, but each cited original must be checked before a final manuscript
  claim.
- **Preprint:** useful efficiency context, never a peer-reviewed reference
  benchmark.

The literature itself cautions against numerical comparison across distinct
experimental settings ([Lee et al., 2022](https://proceedings.mlr.press/v174/lee22a.html)).
That caution is a reason to preserve protocol fields rather than collapse all
papers into a single ranked accuracy column.
