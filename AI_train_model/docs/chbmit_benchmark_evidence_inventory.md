# CHB-MIT Benchmark Evidence Inventory

## Why the Visual Benchmark Has Few Rows

The compact image contains eight rows because it is a **decision table**, not a
literature dump. It retains only the current model, the closest continuous
event-detection comparators, and a small set of classification/hardware
contexts. A single figure containing every historical result would mix three
different tasks and become scientifically misleading.

## Available Evidence

| Evidence source | Extractable CHB-MIT result rows | What the rows contain | Evidence status |
|---|---:|---|---|
| Zhang et al. 2020, P01 Table IV | 8 | Seizure-prediction studies: cases, seizures, features, classifier, SEN, FPR/h, SPH | 7 prior papers are secondary; Zhang is direct |
| Chung et al. 2024, P02 Table 2 and Table 3 | 17 | 13 prior continuous detectors plus 4 Chung channel/annotation configurations; event SEN, FAR/h, delay where reported | Prior-paper rows are secondary; Chung rows are direct |
| Ozkurt et al. 2021, P15 Tables 1-4 | 51 raw rows, 44 unique cited studies | Time, frequency, time-frequency, and nonlinear detection studies; window, subjects, features, classifier, accuracy/SEN/SPE where reported | Secondary evidence; duplicate protocols must be merged carefully |
| Recent direct result papers P04-P09, P12, P19, P30 | 12 local source documents | Modern prediction/detection/classification/efficiency results and model evidence | Direct where the paper reports its own result |

The first three sources alone contain **76 historical result rows** before
cross-source deduplication. That is not 76 directly comparable studies: the
same paper can appear in more than one survey table or report multiple
protocols. The exact number of unique, paper-level comparators must be computed
from a normalized master dataset, using `paper + task + protocol` as the key.

## What Is Usable for Each Claim

| Benchmark output | Include | Exclude |
|---|---|---|
| Main clinical figure | Continuous detection studies with event sensitivity, FAR/h, and delay | Prediction and accuracy-only papers |
| Accuracy/model-size figure | Window-classification studies with declared split, window/channel policy, and model cost | Event-only studies without accuracy; incomparable data integration studies |
| Prediction appendix | Pre-ictal/interictal methods with SEN, FPR/h, and SPH | Ictal seizure detectors |
| Literature master dataset | All rows, tagged `direct`, `secondary`, or `metadata_only` | No rows; provenance remains visible |

## Next Required Artifact

Build `chbmit_benchmark_master.csv` with one row per
`paper + task + protocol + result configuration` and fields for source page,
source tier, patient/seizure coverage, split, channels, window, metrics, and
hardware cost. Deduplicate only after preserving distinct configurations.

The compact visual table should then be generated from this master dataset with
a documented filter, rather than manually selected rows. This is the standard
needed before using the benchmark in a manuscript.
