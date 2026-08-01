# Research Reference Corpus

## Purpose

This directory is the controlled literature index for the KV260 seizure-
detection accelerator study. It covers two linked but distinct evidence
streams:

1. **CHB-MIT / EEG algorithm evidence**: dataset definition, labels,
   preprocessing, model architecture, split protocol, and clinical metrics.
2. **Hardware-aware biomedical AI evidence**: quantization, accelerator
   co-design, FPGA or edge deployment, and how latency, energy, and resources
   must be reported.

The master index is [reference_inventory.csv](reference_inventory.csv). It has
`46` unique entries after removing duplicate rows found in the original
`benchmark.tex`. It is an index of evidence, not a leaderboard.

[authoritative_core.csv](authoritative_core.csv) is the smaller controlled
source set permitted to underpin the current research claims. Its admission,
metric-provenance, seed-reporting, and fair-comparison rules are defined in
[curation_standard.md](curation_standard.md). A high reported accuracy is not
admitted as a comparator until its original evaluation protocol is known.

[tbiocas_core_references.bib](tbiocas_core_references.bib) is a curated
15-source BibTeX seed for the accelerator manuscript. It deliberately contains
only sources likely to support claims in the main text; it is not the complete
literature inventory.

## Evidence levels

| Level | Meaning | Permitted use |
|---|---|---|
| `local_primary` | Local PDF was checked and the article has a source URL/DOI. | Cite and extract method/results after reading the original. |
| `online_primary` | Canonical DOI, publisher, PubMed, or author-hosted primary copy is recorded. | Cite after checking the linked original; download when access permits. |
| `tool_or_standard` | Primary tool documentation or broadly accepted benchmarking/quantization work. | Cite for implementation and measurement methodology. |
| `screening_only` | A candidate found through another paper's comparison table or a preprint. | Do not use for final numerical claims until the original is checked. |

The 22 PDF files already checked are kept in
[`../papers_chbmit`](../papers_chbmit). The hardware-focused retrieval queue is
[`../papers_tbcas_hardware`](../papers_tbcas_hardware). Publisher and repository
access controls prevented automatic PDF downloads for some TBioCAS articles;
their canonical DOI or open repository link is still retained in the index.

## Reading order

### Mandatory before claiming a CHB-MIT result

- `D01` PhysioNet dataset resource and `D02` Shoeb and Guttag for the source
  corpus and detection task.
- `D03` Ali et al. for continuous evaluation, class imbalance, event metrics,
  and cross-subject caveats.
- `A02` Chung et al. and `H02` Busia et al. EEGformer for low-FAR event-level
  evaluation and wearable constraints.

### Mandatory before claiming FPGA deployment

- `H01` Li et al. for seizure-specific hardware/model co-design.
- `H03` Busia et al. SNN FPGA for the closest TBioCAS FPGA comparator.
- `H04` Bahr et al. for deployed model accuracy/latency/energy reporting.
- `M03` Jacob et al. for integer-only inference and `M04` AMD HLS
  co-simulation guidance for verification.
- `M05` MLPerf Tiny for the accuracy-latency-energy measurement principle.

### Architecture rationale and ablations

- `A04`, `A05`, `A13`, and `A14` show raw, DWT, recurrent, and
  depthwise-separable alternatives.
- `A17` and `H02` show why event sensitivity, FAR/h, and delay must remain
  distinct from sampled-window accuracy.
- `H04`, `H07`, and `H08` are lower-power neural alternatives, not direct
  model-size competitors.

## Rules for manuscript tables

1. Keep **detection** and **prediction** in different groups.
2. Never rank accuracy across patient-specific, random-window, within-case,
   and subject-held-out protocols as if they were equal.
3. Treat a neural parameter count as model-only cost. State feature-extraction,
   input buffering, and post-processing boundaries separately.
4. A deployment comparison requires a physical platform and measured or clearly
   labelled estimated latency, energy/power, and resource values.
5. `screening_only` entries may guide literature retrieval, but must not appear
   as a numerical comparator in the submitted paper.

## Relationship to existing project material

- [../../benchmarks/chbmit_ai_hardware_benchmark.csv](../../benchmarks/chbmit_ai_hardware_benchmark.csv)
  is the compact comparison table used during model planning.
- [../papers_chbmit/README.md](../papers_chbmit/README.md) tracks local PDF
  download status.
- [research_argument_map.md](research_argument_map.md) turns the corpus into
  defensible claims for the TBioCAS manuscript.
- [../../paper/tbiocas_kv260/references.bib](../../paper/tbiocas_kv260/references.bib)
  remains the cite-only bibliography; only sources actually used in the paper
  should be added there.
