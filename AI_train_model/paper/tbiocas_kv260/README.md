# TBioCAS Manuscript Draft

This is the active manuscript for the hardware-aware biomedical AI direction:
an integer-only EpiSepNet-5K accelerator on AMD Kria KV260.

## Target And Format

The intended first journal is **IEEE Transactions on Biomedical Circuits and
Systems (TBioCAS)**. Its official guide requires the standard, single-spaced,
double-column IEEE Transactions format, a 100--250 word abstract, at least five
index terms, and normally no more than nine formatted pages. The journal scope
requires a demonstrated synergy between biomedical application and circuits or
systems. Source: [TBioCAS submission guide](https://ieee-cas.org/publication/TBioCAS/tbiocas-manuscript-submission-guide).

The manuscript uses `IEEEtran` and `IEEEtran.bst`, supplied by TeX Live,
Overleaf, or the official [IEEE article templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).

## Build

```bash
latexmk -pdf main.tex
```

This workstation does not currently have a LaTeX distribution, so compilation
must be performed in Overleaf or a TeX Live environment. The expected sequence
is `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

## Evidence Policy

- Results without real KV260 execution remain marked `TBD`.
- The 90.07% FP32 and 90.046% INT16 values are frozen, balanced,
  within-case validation-window results. They are not patient-independent or
  end-to-end FPGA claims.
- The paper must report hardware PPA only after post-route and on-board
  measurements. Keep kernel-only and host-plus-DMA measurements separate.
- Replace all author, affiliation, funding, ethics, and data-availability
  placeholders before submission.

## Current Writing Control

The manuscript skeleton is intentionally ahead of the final results. The
paper's live claim-control matrix and section-by-section writing order are in
[`WRITING_PLAN.md`](WRITING_PLAN.md). The professor-mandated Track-B accuracy
gate is a validation-loss-selected seed-42 result of at least 95.0%, followed
by a three-seed mean of at least 95.0%; five frozen seeds are required before a
final reproducibility result. Until then, the R2 five-second model is a
development candidate only, while the two-second fixed-point reference remains
the engineering baseline.

## Layout

```text
main.tex                 IEEE Transactions manuscript entry point
sections/                Draft narrative, separated by paper section
tables/                  Source-controlled result and template tables
figures/                 Final vector figures only; see figures/README.md
references.bib           IEEE-style bibliography database
```

The experiment and evidence plan is maintained in
[`../../docs/hardware_aware_research_verification.md`](../../docs/hardware_aware_research_verification.md).
