# Candidate Registry V2.1 Addendum

Registry V2.0 declared six baseline families but did not freeze their exact
implementation settings. V2.1 adds those settings before any baseline is
trained and before any V2 outer future partition is opened. P1 and P2 results
remain V2.0 historical-reference evidence; their artifacts are not rewritten.

The baseline implementation mappings are stored directly in
`configs/candidate_registry_v2.json`. Every new baseline provenance record
contains the candidate ID and SHA-256 hash of this registry file.

This is an inner-development protocol clarification, not a model promotion or
an outer-test analysis. Candidate changes after any V2 outer-fold result
requires a new registry version and explicit disclosure.
