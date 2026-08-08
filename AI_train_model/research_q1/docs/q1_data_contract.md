# Q1 G1 Data Contract

G1 is a read-only audit of a pre-existing CHB-MIT snapshot. Runtime access is
allowed only through `CHBMIT_RAW_DIR`; raw EDF files, waveform arrays and
prepared data are never copied into this repository.

`RECORDS` is the canonical file inventory. `RECORDS-WITH-SEIZURES`, case
summary metadata, optional machine-readable seizure annotations and
`SHA256SUMS.txt` are independently audited. A conflict is reported as an audit
failure, never repaired or silently resolved.

Each recording is a separate temporal object. The canonical manifest preserves
`dataset -> subject_id -> case_id -> session_id -> recording_id -> relative_path`
and relative recording time. It contains no train/validation/test split and no
montage decision. `chb01` and `chb21` share biological `subject_id` and
`split_group`, while retaining distinct case and session identities.

The 2026-08-08 snapshot note is descriptive only: current verified manifests
were observed to contain 686 records and 142 seizure-containing records,
whereas older prose may state 664 and 129. G1 derives all values from the
runtime manifests and checksum snapshot.
