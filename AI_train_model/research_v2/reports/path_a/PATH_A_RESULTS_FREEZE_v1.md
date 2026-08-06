# Path A Results Freeze v1

**Status:** Accuracy ladder frozen for reporting (2026-08-06).  
**Branch:** `research/patient-specific-a1-v1`  
**Best configuration:** **A1.2** (hierarchical 31/7/3 + train-only SupCon 0.05 / T=0.1)  
**Primary metric:** Unweighted mean of per-case **sealed test balanced window accuracy** (threshold 0.5)

Machine-readable cohort tables:

| File | Content |
|---|---|
| `ps_a1_cohort_test_summary.json` | A1.0 baseline @ thr 0.5 |
| `ps_a1_valthr_cohort_summary.json` | A1.0b val-selected threshold (rejected) |
| `ps_a11_cohort_test_summary.json` | A1.1 multiscale (rejected) |
| `ps_a12_cohort_test_summary.json` | **A1.2 SupCon (best)** |

---

## 1. Headline numbers

| Protocol step | Mean test balanced | SD | N | Decision |
|---|---:|---:|---:|---|
| Shared clean-slate A0 (3 seeds, one model) | 88.030% | 1.14% | 3 seeds | Shared baseline (different claim) |
| **A1.0** hierarchical 31/7/3, thr 0.5 | 90.719% | 12.84% | 24 cases | Patient-specific baseline |
| A1.0b val-selected threshold | 89.918% | 12.33% | 24 | **Rejected** (worse than 0.5) |
| A1.1 compact MSR ≤25k, thr 0.5 | 88.120% | 15.19% | 24 | **Rejected** (Δ −2.60 pp vs A1.0) |
| **A1.2** A1.0 + SupCon 0.05/T0.1, thr 0.5 | **91.096%** | **12.68%** | **24** | **Best Path A mean** (Δ +0.38 pp vs A1.0) |
| Target (predeclared) | ≥ 95.0% mean | — | 24 | **Not met** (−3.90 pp) |

```text
PATH A BEST (frozen for report):
  mean_test_balanced = 91.096% ± 12.68%  (N=24 cases, seed 42)
  model = hierarchical_separable_1dcnn 31/7/3 (~4,917 params) + SupCon train-only
  threshold = 0.5
  cohort success (≥95% mean) = FAIL
```

---

## 2. Allowed claims (wording)

**Allowed**

> Under a patient-specific chronological CHB-MIT protocol (one model per case,
> 60/20/20 recording split within case, raw 17-channel 5 s windows, causal IIR,
> train-only z-score), a compact hierarchical separable 1D-CNN with train-only
> supervised contrastive loss achieved a **mean sealed-test balanced window
> accuracy of 91.096% ± 12.68% across 24 eligible cases** (seed 42, threshold 0.5).
> Approximately **14/24** cases reached ≥95% balanced test accuracy; several
> cases approached 100%.

**Not allowed**

- “Achieved 95% on CHB-MIT” (cohort mean target failed)
- Ranking against LMPSeizNet / Wang / Kashefi without protocol labels
- Dropping chb16/chb18/chb24 to raise the mean without a predeclared rule
- Calling A1.1 or A1.0b the best configuration
- Claiming patient-independent or external clinical validation

---

## 3. Per-case pattern (A1.2 best)

| Band | Approx. count | Role |
|---|---:|---|
| ≥ 95% balanced test | ~14 / 24 | Strong personalization (Wang-style high end) |
| 90–95% | ~7 / 24 | Near target |
| Hard failures | **3** | **chb16 ~77%**, **chb18 50%**, **chb24 ~56%** dominate SD and block 95% mean |

Hard-case sketch:

- **chb18:** balanced ~50%, sensitivity ~100% → over-calling seizure at thr 0.5  
- **chb24:** balanced ~56%, sensitivity ~12%, AUROC very low → weak ranking / collapse  
- **chb16:** balanced ~77%, high sensitivity, weak specificity on balanced set  

---

## 4. Ladder decisions (frozen)

| ID | Change | Outcome |
|---|---|---|
| A1.0 | Hierarchical 31/7/3 | Keep as architecture family |
| A1.0b | Val-max balanced threshold → test | Reject transfer failure |
| A1.1 | Compact multiscale residual ≤25k | Reject (mean regression) |
| A1.2 | SupCon train-only on A1.0 graph | **Promote as best mean** (+0.38 pp) |
| A1.2b / A1.3 / hard-case only | Optional future work | **Not required** for this freeze |

Accuracy Path A is **recorded**. Further training is optional research, not required to close this freeze document.

---

## 5. Relation to shared clean-slate

| Track | Claim | Best sealed test |
|---|---|---|
| Clean-slate shared | One model, all patients, chronological 60/20/20 **across** corpus | **88.03% ± 1.14%** balanced (3 seeds) |
| Path A patient-specific | One model **per case** | **91.10% ± 12.68%** mean over cases |

Report **separately**. Do not average them into one number.

---

## 6. Source artifacts (server-side names)

| Stage | Path pattern |
|---|---|
| Per-case protocol | `data/chbmit_protocol_ps_a1_v1/{case}/` |
| Per-case prepared NPZ | `data/chbmit_prepared_ps_a1_v1/{case}/` |
| A1.0 train / test | `outputs/ps_a1_{case}_s42/`, `outputs/ps_a1_test_ps_a1_{case}_s42/` |
| A1.2 train / test | `outputs/ps_a12_{case}_s42/`, `outputs/ps_a12_test_ps_a12_{case}_s42/` |
| Tracked summaries | `research_v2/reports/path_a/*.json` |

Large `outputs/**/*.pth` and `data/**/*.npz` remain gitignored; only summaries are versioned.
