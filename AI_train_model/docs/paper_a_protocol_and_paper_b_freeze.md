# Research State: Paper A Active, Paper B Frozen

## Decision on 1 August 2026

The programme now has two independent papers. Their model identities, metrics,
and claims must not be mixed.

| Paper | Scope | Status | Frozen reference / active target |
|---|---|---|---|
| Paper B: hardware-aware biomedical AI / KV260 | INT16 tensor export, HLS implementation, KV260 synthesis and board measurement | **Frozen until hardware work resumes** | EpiSepNet-5K: 17 x 512 input, separable 1D-CNN (31/15), 5,013 trainable parameters, `run_21_raw_2s_temporal3` |
| Paper A: accuracy-focused seizure detection | Raw EEG 1D-CNN architecture study with fewer than 100K trainable parameters | **Active** | EpiSepNet-A-MSR development candidate, five-second input and patient-aware final evaluation |

Paper B must not receive an architecture, threshold, preprocessing, or
checkpoint change during Paper A development. Its frozen FP32 and INT16
artifacts remain deployment references only; no KV260 latency, power, or
resource claim is made before toolchain synthesis and board measurement.

## Paper A question and constraint

**Question:** can a raw-EEG, CNN-only model with fewer than 100K parameters
reach at least 95% *balanced window accuracy* under a predeclared temporal
test protocol, while retaining useful continuous event behaviour?

The model graph may use only Conv1D (including depthwise and pointwise Conv1D),
BatchNorm, ReLU, pooling, residual addition, dropout, and a linear classifier.
It must not use an RNN, LSTM/GRU, Transformer, attention block, GAN, or
handcrafted spectral/DWT frontend as its primary candidate.

The `paper_a_multiscale_residual_1dcnn` candidate is therefore a valid Paper A
model: parallel short/long depthwise temporal paths, pointwise channel mixing,
and residual depthwise-separable temporal blocks. The constructor rejects a
configuration over 100,000 parameters.

## Metric contract

The natural CHB-MIT test ratio has approximately one ictal window per ten
non-seizure windows. A trivial all-nonseizure classifier consequently achieves
about 90.9% raw accuracy. Paper A must therefore report all of:

1. Primary: balanced window accuracy, sensitivity, precision, F1, AUROC and
   average precision at a threshold fixed on validation data.
2. Secondary: natural-prevalence test accuracy and confusion matrix.
3. Clinical context: continuous event sensitivity, FAR/h, and detection delay;
   all alarm threshold/persistence settings are selected on validation only.

The earlier `run_84` and `run_85` test probes are exploratory and cannot be
used to select this model, its preprocessing, or its hyperparameters.

## Evaluation plan

### Development

- Retain the audited 17-channel montage, raw signal representation, 256 Hz,
  five-second windows and one-second stride.
- Fit filtering and z-score statistics on the training partition only.
- Run all architecture and optimisation screens on training/validation data;
  use a 50-epoch cap, early-stopping patience 6, and the validation-loss
  checkpoint rule established by the R2 study.
- Each screen starts at seed 42. A candidate that reaches 95% balanced
  validation accuracy receives seeds 7 and 123; the selected candidate then
  receives seeds 314 and 2718 for a five-seed result.

### Formal Paper A result

Before formal evaluation, freeze one architecture, preprocessing contract,
training schedule and alarm policy. Then run a patient-group-disjoint outer
evaluation or nested grouped cross-validation. `chb01` and `chb21` remain one
participant. The group-disjoint result is a generalisation analysis, not a
numerical substitute for personalized temporal testing.

The prior R2 probes opened the original CHB-MIT test partition. A reshuffle of
the same corpus must **not** be described as a newly blinded test. The formal
CHB-MIT result is consequently an internally validated, predeclared nested
grouped-CV result with five-seed uncertainty. A truly independent test claim
requires an external cohort or a previously unused, access-controlled cohort.

The 95% target applies to predeclared held-out chronological folds for
personalized detection and must be described as within-patient temporal
generalisation. If the patient-group-disjoint result is lower, report it
separately rather than presenting it as a failure or merging the protocols.

## Ordered Paper A experiments

| Stage | Candidate / change | Selection purpose |
|---|---|---|
| A0 | R2 47/7/3, 5,733 parameters | Reproducible five-second baseline; no more use of exposed test split |
| A1 | EpiSepNet-A-MSR, 57,446 parameters by construction | Test multiscale morphology plus residual context, not width alone |
| A2 | Same topology at roughly 25K, 50K and 80-95K parameters | Establish capacity-performance curve under the fixed 100K ceiling |
| A3 | One causal filtering / train-only normalization ablation on the winner | Quantify deployable preprocessing cost |
| A4 | Five training seeds and frozen protocol | Confirm variance before formal evaluation |
| A5 | Patient-group outer evaluation and continuous event evaluation | State generalisation and clinical limitations honestly |

No test set is opened at A0-A4. A rejected configuration remains a recorded
negative ablation.

## Research basis

High compact-CNN results exist but their protocols are heterogeneous.
LMPSeizNet reports 97.42% with 18,024 parameters under epoch-level CV, whereas
PSD-LW-DCN reports 85.84% with 61,218 parameters under cross-subject LOSO.
These results support a sub-100K 1D-CNN capacity study but do not justify a
direct numerical comparison. Continuous cross-subject work also shows that
event behaviour is materially harder than balanced-window classification.

- Alsaadan et al., [LMPSeizNet](https://doi.org/10.3390/math12233648)
- Gu et al., [PSD-LW-DCN](https://doi.org/10.1038/s41598-026-44536-y)
- Ali et al., [CHB-MIT overlooked perspectives](https://doi.org/10.1098/rsos.230601)
- Adatia et al., [efficient multichannel 1D-CNN](https://doi.org/10.1109/EMBC58623.2025.11254246)
