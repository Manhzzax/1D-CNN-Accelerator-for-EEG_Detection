# TBioCAS And Hardware-Focused Retrieval Queue

This directory is reserved for legally accessible primary PDFs that directly
support the KV260 paper. The master metadata and canonical links are in
[`../reference_corpus/reference_inventory.csv`](../reference_corpus/reference_inventory.csv).

## High-priority direct comparators

| ID | Paper | Why it matters | Canonical access |
|---|---|---|---|
| H01 | Li et al., 2022, parallel memristive CNN | Seizure-specific model-hardware co-design in TBioCAS. A verified preprint is already `../papers_chbmit/09_li_2022_parallel_memristive_cnn_detection_prediction.pdf`. | https://doi.org/10.1109/TBCAS.2022.3185584 |
| H02 | Busia et al., 2024, EEGformer | Raw EEG, low channel count, FAR/delay, and measured MCU latency/energy. | https://doi.org/10.1109/TBCAS.2024.3357509 |
| H03 | Busia et al., 2025, FPGA SNN | Closest published TBioCAS FPGA seizure-monitoring comparator. | https://doi.org/10.1109/TBCAS.2025.3575327 |
| H04 | Bahr et al., 2021, GAP8 CNN | Deployed CHB-MIT CNN with measured accuracy, latency, and energy. | https://doi.org/10.3390/bios11070203 |
| H05 | Alhammadi et al., 2022, 1D-CNN HLS accelerator | Conv1D/HLS implementation rationale. A verified PDF is already `../papers_chbmit/25_alhammadi_2022_1dcnn_fpga_accelerator_hls.pdf`. | Local verified PDF |

## Retrieval policy

- Store only lawful open-access, author-accepted, or user-provided PDFs.
- Verify every downloaded file starts with `%PDF` before marking it local.
- Keep DOI/publisher links even when an automated request is rejected. Do not
  substitute a search-result snippet for the original paper.
- Record a new file in `reference_inventory.csv` and update this README when a
  PDF becomes available.
