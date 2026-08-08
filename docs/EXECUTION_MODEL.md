# G1A execution model

## Trust boundary

- **Codex/local** develops source code and runs synthetic tests only. It has
  no access to SERVER-02 or the CHB-MIT raw snapshot and must not report a real
  audit, checksum verification, or dataset conformance result.
- **GitHub `main`** is the public source of truth and control plane. Review
  happens through a feature-branch pull request; only an approved commit SHA is
  eligible for server execution.
- **SERVER-02** is the data plane. Its operator checks out the exact approved
  SHA in detached HEAD state, sets `CHBMIT_RAW_DIR`, and runs the read-only
  preflight and then the audit.

## Data handling

`CHBMIT_RAW_DIR` is required at runtime and has no fallback. The raw snapshot
is read-only. EDF signal samples are never loaded by G1A. Raw EDF files,
prepared arrays, caches, and checkpoints are server-only and are not copied or
committed. G1A's shareable provenance deliberately omits absolute paths,
hostnames, and usernames.

## SERVER-02 prerequisites

Before using the single audit command, the server operator verifies Python
3.11 or newer, installs the declared project dependencies through the approved
server environment, and confirms `pyedflib` and `wfdb` import successfully.
Codex and the audit script do not install packages or otherwise modify the
server environment.

## Safe handoff

1. Codex pushes a feature branch and opens a PR into `main`.
2. Reviewers approve and merge it. Record the resulting approved commit SHA.
3. The SERVER-02 operator confirms a clean worktree, fetches GitHub, and
   checks out that SHA exactly — never an arbitrary branch state.
4. The operator runs the single documented `scripts/run_g1_audit.sh` command.
   It runs synthetic G1 tests, then `eegkv preflight-g1` (JSON, no files), then
   the full checksum audit. A failure is returned for investigation, not
   repaired automatically.
5. Generated small artifacts can be reviewed and committed separately after
   inspection.

G1A does not authorize G1B, training, quantization, HLS, or hardware work.
