# TBioCAS Writing Plan and Claim-Control Matrix

## Manuscript Status

The LaTeX skeleton in this directory is built: it has an IEEE Transactions
entry point, abstract, introduction, system method, accelerator methodology,
experimental protocol, results, discussion, conclusion, bibliography, and
tables. It is **not submission-ready**. The existing text describes the frozen
two-second EpiSepNet-5K software reference; all KV260 PPA statements remain
`TBD` by design.

The paper must not be completed as a retrospective narrative around the best
number. It is written in two stages:

1. **Now:** write stable methods, dataset audit, model contract, integer/HLS
   verification plan, literature context, and limitations.
2. **After gates:** insert frozen accuracy, causal event, patient-held-out,
   integer, and KV260 measurements. Write the abstract, result claims, and
   conclusion last from those immutable artifacts.

## Intended Paper Thesis

> A raw multichannel depthwise-separable 1-D CNN can be co-designed with an
> integer-verified KV260 implementation, and a controlled input-context study
> quantifies the accuracy versus buffering, latency, resource, and energy
> trade-off.

This is a hardware-aware biomedical AI claim. It is not a universal CHB-MIT
accuracy ranking, medical-device claim, or external clinical-validation claim.

## Mandatory Gates Before a Headline Result

| Gate | Required result | What it permits |
|---|---|---|
| G1: accuracy screen | Validation-loss-selected seed 42 >=95.0% balanced validation accuracy | Replicate the candidate; no paper headline yet |
| G2: accuracy confirmation | Mean >=95.0% across seeds 42, 7, 123 | Promote candidate to frozen-accuracy winner |
| G3: stability | Winner and R2 baseline each evaluated on five predeclared seeds | Mean +/- SD and reproducibility claim |
| G4: causal/event | Causal-IIR continuous validation with frozen threshold/policy, then patient-group-held-out test | Event sensitivity, FAR/h, delay, generalisation scope |
| G5: integer | FP32 -> integer reference -> RTL agreement with declared arithmetic | Fixed-point fidelity claim |
| G6: KV260 | Post-route and on-board resources, latency, throughput, power, energy | Accelerator/PPA claim |

## Claim-Control Matrix

| Paper claim | Evidence already available | Evidence still required | Wording allowed now |
|---|---|---|---|
| CHB-MIT data provenance | 686 EDF, 198 primary intervals, 17 canonical bipolar channels, audit artifacts | Cite annotation precedence and one documented interval discrepancy | "Audited CHB-MIT corpus" |
| Compact raw CNN | Frozen 2 s 5,013-parameter graph; R2 5 s 4,917-parameter graph | MAC and activation-memory reports for selected model | "Compact at weight level" |
| Accuracy improvement | R2 5 s: 93.081% +/- 1.096% across three seeds | >=95% gate, five-seed replication | "Development candidate", not final result |
| Reproducible training | Seeds, hyperparameters, checkpoints, locked artifacts | Frozen five-seed ledger and hashes | "Three-seed development replication" |
| Clinical detection | None for frozen R2 5 s candidate | Causal continuous event results and patient-group test | No clinical-performance claim |
| Quantization fidelity | 2 s reference INT16 emulator agreement | Selected winner's integer reference and RTL agreement | "Software fixed-point reference" for 2 s only |
| KV260 efficiency | Board target and planned HLS flow | Post-route/on-board PPA and power method | No FPGA-speed, energy, or resource claim |

## Issues That Must Be Disclosed

1. **Within-case development split:** same patient can appear in train and
   validation recordings. It is not patient-independent evaluation.
2. **Balanced sampled windows:** 50/50 ictal/interictal accuracy does not
   represent the natural prevalence or false alarms during continuous EEG.
3. **Five-second full-ictal windows:** this changes the positive population
   from the two-second experiment and excludes ambiguous onset/offset windows.
4. **Window dependence:** 5 s windows with 1 s stride overlap by 80%; use
   patient/recording block uncertainty, not independent-window confidence
   intervals.
5. **Preprocessing causality:** `zero_phase` is offline. Streaming claims
   require a causal-IIR replication and a saved preparation artifact.
6. **Development-selection risk:** all architecture/hyperparameter screens use
   validation only; historical observed tests are never final paper tests.
7. **Seed interpretation:** seed values are arbitrary PRNG states, not years,
   patient IDs, or folds. Report a predeclared five-seed set and all outcomes.
8. **Parameter count is incomplete hardware cost:** input, activations, DMA,
   preprocessing, and dataflow must accompany weights in the memory analysis.
9. **Literature comparability:** detection/prediction, split, channels,
   labels, window duration, feature frontend, and class ratio differ between
   papers. Tables are context, not a global leaderboard.
10. **CHB-MIT scope:** one public pediatric corpus does not establish external
    clinical validity, safety, or medical-device utility.

## Section-by-Section Writing Method

| Section | Write now | Populate only after frozen evidence |
|---|---|---|
| Abstract | Problem, raw Conv1D/KV260 scope, method boundary | Final accuracy, event, quantization, PPA, and energy numbers |
| Introduction | Clinical motivation, protocol heterogeneity, hardware gap, contribution hypothesis | No state-of-the-art claim |
| System method | Channels, window, model equations, design-point definitions, memory/MAC method | Selected winner identity only after G2 |
| Accelerator method | Integer arithmetic, HLS variants, DMA boundary, measurement protocol | Exact directives/clock/implementation version after G6 |
| Experimental protocol | Audit, splits, labels, normalisation, seed protocol, metrics, statistical plan | Final test only after G4 |
| Results | Tables that are already factual and marked development-only | G1--G6 result tables and figures |
| Discussion | Trade-off interpretation and all limitations | Comparison of measured 2 s/5 s PPA and failure analysis |
| Conclusion | Scope-limited contribution | Final conclusion from frozen results only |

## Required Tables and Figures

1. **Table I:** dataset audit, channels, labels, split, window/stride, class
   ratio, filter, and normalisation contract.
2. **Table II:** accuracy development ledger: baseline, every controlled
   ablation, five frozen seeds, parameters, MACs, input and activation memory.
3. **Table III:** continuous causal patient-group-held-out event results:
   sensitivity, FAR/h, delay, per-patient range, and confidence interval.
4. **Table IV:** quantization fidelity: FP32, integer C, RTL, and board
   agreement using the same frozen samples.
5. **Table V:** KV260 PPA: clock, LUT, FF, BRAM, URAM, DSP, latency,
   throughput, power, and energy/window for measured variants.
6. **Figure 1:** end-to-end system boundary from EEG tensor to KV260 output;
   label filtering/acquisition as outside the initial accelerator boundary.
7. **Figure 2:** model/operator and dataflow diagram with tensor shapes.
8. **Figure 3:** accuracy versus buffer/MAC/energy trade-off for measured 2 s
   and 5 s design points, not a cross-paper leaderboard.
9. **Figure 4:** event detection timeline and per-patient FAR/sensitivity plot.

## Writing Rules

- Cite primary sources for any literature number; mark secondary rows as
  context only until verified from the original paper.
- Separate window metrics, event metrics, and board metrics in prose and
  tables. Never use one in place of another.
- Use `mean +/- SD` across seed trials and name the exact seed set.
- Write result prose from tracked JSON/CSV artifacts, never copied terminal
  output or manually transcribed values.
- Keep `TBD` markers until the corresponding gate is complete. A missing
  measurement is not filled with an estimate.

## Current Editing Order

1. Retain the existing 2 s fixed-point/HLS draft as the engineering baseline.
2. Add the accuracy-development protocol and 5 s R2 design-point definition to
   the Methods section, but label it development-only until G2/G3.
3. After the >=95% winner is frozen, replace the generic model language with
   the exact architecture/seed/artifact identifiers.
4. After G4--G6, complete Results, Discussion, Abstract, and Conclusion in
   that order, then remove every `TBD` marker.
