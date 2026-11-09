# Changelog

## 0.1.0 (unreleased)

First version.

- `Request` / `PackedRequest` and the `/v1/systemone` wire format (`choice` and `noul` questions).
- `JevBackend` (pinned model, retry with backoff, on-disk response cache) and an offline `MockBackend`.
- Information levers: retrieved examples, definitions, neutral-id renaming, stated class shares, post-hoc share
  correction, packing several questions about one state.
- Local models: similarity-weighted kNN vote, tuned prior-corrected logistic regression, product-of-experts
  `combine`.
- `ClassifierFirstCascade` with accuracy or rare-class recall targets.
- Labels-only shortlist.
- Evaluation: near-duplicate-aware holdout split, cluster bootstrap, TOST, Holm, A/A null controls, promotion floor,
  `compare`, `freeze`.
- Ten runnable examples on a bundled toy dataset; `docs/algorithms.md`, `docs/what_not_to_do.md`.
