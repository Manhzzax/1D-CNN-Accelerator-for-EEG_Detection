# Q1 errata and staged conformance

## Snapshot manifests

The current official snapshot manifests contain **686 `RECORDS`** and
**141 `RECORDS-WITH-SEIZURES`** entries. Earlier descriptive CHB-MIT prose and
publications may instead report 664 recordings and 129 seizure-containing
recordings. G1A derives its observed counts from the pinned local `RECORDS`
and `RECORDS-WITH-SEIZURES` files and verifies their binding through
`SHA256SUMS.txt`; the prose figures are not executable constraints. The
portable snapshot identifier is fixed as
`chbmit-1.0.0-physionet-wget-20260728` for the declared wget snapshot.

## Stage separation

- **G1A** is the snapshot audit: inventory, checksums, EDF headers, official
  annotations, identity mapping, channel census, and portable provenance.
- **G1B** is later montage/scorer conformance. It may begin only after the
  real G1A artifacts have been reviewed and approved.

## Montage proposal is not a decision

An 18-channel double-banana adapter is a pending proposal only. It is not
implemented, selected, or used by G1A. The real corpus-wide channel census must
confirm its feasibility before G1B can specify a montage.
