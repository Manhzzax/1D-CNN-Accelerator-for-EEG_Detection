# Project Contract v1

This clean-room repository implements the Q1 research specification. It has
no scientific dependency on the repository history that preceded its orphan
`main` branch.

## Active G1A protocol

- Primary dataset: independently acquired CHB-MIT EDF corpus.
- Participant identity: `chb01` and `chb21` are one participant group.
- `RECORDS`, `RECORDS-WITH-SEIZURES` and `SHA256SUMS.txt` are snapshot source
  truth. G1A derives counts from those manifests rather than historical prose.
- Every EDF remains a separate temporal object. Canonical identities are
  relative and preserve dataset, subject, case, session and recording.
- `chb01` and `chb21` share `subject_01_21`; no other cross-case identity is
  inferred.
- G1A audits raw channel labels and patterns but makes no 17/18/19-channel
  choice. It creates no split, windows, normalization state or model input.

G1B montage and scorer conformance remains blocked until SERVER-02 has run an
approved G1A commit against the real snapshot and its artifacts are reviewed.

Existing G2 source is retained for later review but its execution commands are
intentionally unavailable until G1 is approved. No G1 artifact may include an
absolute server path in its canonical manifest.
