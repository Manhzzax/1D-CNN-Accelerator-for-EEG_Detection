# Authoritative Reference Curation Standard

## Scope

This standard governs the evidence used in the two project tracks:

- **Paper A:** compact 1D-CNN seizure detection and clinically meaningful
  evaluation.
- **Paper B:** hardware-aware biomedical AI and the KV260 accelerator.

The controlled shortlist is [authoritative_core.csv](authoritative_core.csv).
The broader discovery record remains [reference_inventory.csv](reference_inventory.csv).
An item in the inventory is not automatically eligible for a manuscript claim.

## Admission rules

An item is in the authoritative core only when it is one of the following:

1. A primary peer-reviewed article from its publisher, PubMed Central,
   PhysioNet, or an author-hosted accepted manuscript.
2. Official dataset or vendor documentation needed to reproduce the study.
3. A peer-reviewed review used only for background or literature mapping.

Preprints, ResearchGate-only copies, comparison tables from a review, and
unverified web summaries may suggest papers to retrieve but cannot support a
headline numerical claim. Detection and prediction are always separate.

## Mandatory extraction fields

Before a paper enters a benchmark table, record all fields below. Use `NR`
when the original article does not report the field.

| Field | Why it is required |
|---|---|
| Task and label definition | Detection, onset detection, and prediction have different endpoints. |
| Cohort and EDF/case count | CHB-MIT coverage varies substantially. |
| Input and representation | Raw EEG, PSD, DWT, STFT, CSP, and images have different cost boundaries. |
| Channels, window length, stride | Directly affect accuracy, latency, input memory, and accelerator cost. |
| Split unit and patient isolation | Prevents ranking window, recording, patient-specific, and patient-held-out results together. |
| Metric provenance | Explicitly state `validation`, `held-out CV fold`, `internal test`, or `external test`. |
| Class prevalence and sampling | Raw accuracy is misleading on long interictal EEG. |
| Training seed and repetitions | Separates stochastic replication from split/fold variation. |
| Model parameters and precision | Model cost is distinct from checkpoint size and feature extraction. |
| Event sensitivity, FAR/h, delay | Required for continuous clinical-monitoring interpretation. |
| Hardware boundary and measurements | Required for Paper B latency, energy, and KV260-resource claims. |

## Evidence labels

| Label | Meaning | Manuscript use |
|---|---|---|
| `verified_primary` | Original full text checked and mandatory fields extracted. | May support a direct qualified claim. |
| `primary_pending_extraction` | Publisher DOI or canonical primary record checked, but full protocol is not yet extracted. | Background or retrieval target only; no numerical ranking. |
| `local_primary` | Local PDF exists but extraction remains incomplete. | Cite for a checked statement only. |
| `review_only` | Review or survey. | Background and paper discovery, not primary numerical evidence. |
| `screening_only` | Preprint, secondary citation, or unverified copy. | Not for submitted-paper numerical claims. |

## Fair benchmark rules

1. A validation score from this project is labelled `validation accuracy`; it
   is never compared as a final test score.
2. A cross-validation value is labelled `mean held-out fold score`, including
   whether the held-out unit is a window, recording, or patient.
3. A patient-held-out outer fold is labelled `outer-CV test score`. It remains
   distinct from an untouched external cohort.
4. An exposed test partition cannot be reused as a final selection benchmark.
5. Report `NR` for missing seeds, repetitions, or parameter counts. Never
   infer them from architecture figures or a model-file size.
6. For Paper A, compare accuracy only inside a protocol family. For Paper B,
   also disclose precision, platform, clock, measurement boundary, resources,
   latency, power, and energy.

## Project standard for new experiments

Paper A development may use the locked within-case chronological validation
split for architecture screening. Its final claim requires a frozen model and
predeclared patient-group outer evaluation, with all normalization and model
selection restricted to the inner training data. Report five fixed training
seeds and mean plus standard deviation.

Paper B uses the frozen reference model. The KV260 evidence chain is: exported
tensors and fixed-point contract, C reference, C/RTL agreement, post-route
resource and timing reports, then board-level latency, power, energy and
FP32/INT16 agreement. Hardware measurements must not trigger AI model tuning.

## Reading order

1. `D01`, `D02`, and `D03` establish dataset, labels, and event evaluation.
2. `A02`, `A04`, `A05`, `A13`, and `A16` establish detection-model trade-offs.
3. `M01` and `M03` establish separable-CNN and integer-inference rationale.
4. `H01` through `H05`, `T01`, and `T02` establish the TBioCAS hardware
   evidence standard.
