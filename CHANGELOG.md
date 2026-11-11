# Changelog

## 0.3.0 (unreleased): synced with paper v26

Synced with paper v26, "Rules of Thumb for Decision Models: What Makes Performance Better (or Worse)". Every number
in the README and docs is quoted from v26, apart from the package constant `PAPER_EXPONENTS`. Code behaviour is
unchanged: only docstrings and comments were edited.

- **v25 and v26 revisions** (first synced at v24): "ten planned runs" (eight public, two private); the one-rival
  penalty on 6 of 6 others on fresh items and 5 of 6 on private data (Kev-4B reversed); Do 5 is "local match not
  shown"; leakage is "unlikely" (rank correlations 0.76 to 1.00, 0.49 without the two largest effects; the planned
  distortion test flagged two techniques); descriptions scoped to two tool catalogues; status words defined in
  plain words; "bare call" throughout; and the paper's plainer wording ("three-example test", "unrelated
  questions", "on the central estimate", "an attack search that never saw the scores").

**Docs**

- README rewritten as a practitioner page for v26: a plain summary; "How a call works" (the paper's Figure 2, with a
  diagram of the request, where information goes and the optional pool step); the six do's and six don'ts of Table
  2 with evidence and status, each mapped to the function or example that implements it, or marked "guidance
  only"; "Which approach?" by label budget (Section 5: use the prompt at 5 per class; at 100 per class prefer it
  unless privacy rules it out); "Results across models" (Section 6) and the leakage check on four private datasets
  (Section 7), with the caveat that examples misled two models on long texts; "What didn't work" (Table 7); cost
  from the paper's Figure 4; model versions, now including Jev 1.14 and the other models the paper tested.
- Status words follow the paper: confirmed, observed, secondary, not shown. "Pre-registered" and "exploratory" are
  replaced by "planned in advance" and "observed".
- `docs/what_not_to_do.md` rewritten and shortened: pool noise and the one-rival penalty across models, placement,
  stale shares, names without meaning (the names and descriptions experiment), the long-text failure mode and the
  leakage check.
- `docs/why_it_works.md` updated: the prior and evidence idea in plain words, the Appendix I.13 model with v26's
  numbers (ε = 0.0225), placement, other models, the long-text failure mode and the leakage check.
- `docs/algorithms.md`: the cluster note in the paired bootstrap now matches v26 (gold classes in Runs 3 to 8, and
  the class bootstrap is too liberal with few classes); the tempered product is labelled observed.
- `examples/real_payloads/README.md` and `examples/12`: the worked example is the paper's Figure 3.

**Docstrings and comments (text only)**

- Package description and `__init__` docstring use the new title; version 0.3.0.
- `combine`, `local.py`, `examples/05`: the local combination trailed the prompt by 0.7 points in both fresh-item
  tests; it is no longer described as non-inferior.
- `rename_to_ids`, `examples/03`: ids with a description are one of several naming options, not the headline;
  readable names with a one-line description are the default.
- `examples/02`, `04`, `06`, `07`, `08`, `10`, `shortlist.py`, `levers.py` (wrapper and packing docstrings),
  `evaluate.holm`, `PAPER_MODEL` comment: claims and status words matched to v26.

## 0.2.0 (unreleased)

Synced with the paper's v11: Run 3 results, corrected numbers, and the repo critique of the practitioner review.
Then synced with v12: Run 4 results.

**Run 4 (paper v12)**

- README: the Run 4 rows of the results table. X3a-c confirmed (deletion costs less than relabelling; one common
  rival costs more than distinct rivals; deletion jumps at the last correct example). X1a and X1b not shown
  non-inferior on MASSIVE, LEDGAR and MetaTool. Secondary: tempered product −0.4, SuperICL below the prompt, pool
  redraws (G5's sign not stable), and the CFPB time split (no gain from retrieval with a 2025 natural-share pool on
  2026 items).
- Decision guide matched to the paper's Section 9: the bare call × centroid tip is dropped, the 5-per-class product
  is "not established", the 100-per-class branch adds Run 4 and the pool redraws, a time-split caution is added,
  wrong labels are explained as lost correct evidence plus rival evidence (C4), and "confirm before you ship" notes
  the paper's test sizes.
- `docs/why_it_works.md`: Run 4's out-of-sample check of the fitted model. `docs/what_not_to_do.md`: X3a-c and the
  time split.

**Added**

- `ClassifierFirstCascade.fit(..., cv=k, groups=..., repeats=...)`: cross-fitted threshold selection. The threshold
  is chosen on k-1 folds of the calibration items and applied to the held-out fold (whole near-duplicate groups per
  fold); `report["crossfit"]` holds the out-of-fold share sent, accuracy, recall and precision and whether the
  target is met out of fold. The in-sample report is now flagged `optimistic`.
- `ClassifierFirstCascade.fit(..., min_precision=..., positive=...)`: a minimum-precision target for a rare class,
  alone or together with `min_recall`. The report gives the rare class's recall and precision, and the full model's.
- `combine(..., weights=(a, b, c))`: the tempered product of experts; `weights="paper"` and `tempered_combine` use
  `PAPER_EXPONENTS = (0.3579, 0.1310, 0.6201)`, the paper's frozen exponents at about 100 labels per class.
  `combine` without `weights` is unchanged (the plain product of G5).
- `fit_exponents(bare, [votes, lrs], gold, cv=5, groups=...)`: fits the exponents by cross-validation (gold log-loss,
  exponents >= 0) on your own held-out items and reports the out-of-fold log-loss and accuracy next to the plain
  product's.
- `decision_levers.data.load_csv(path, text_col, label_col, where=...)` and `LabelledPool.from_csv`.
- `evaluate.class_metrics`: accuracy, macro-F1, per-class recall and worst-class recall.
- `ReplayBackend` (answers byte-identical recorded bodies offline) and `Backend.decide_wire` (send a raw body).
- `examples/11_your_own_data.py` with a 60-row hand-written CSV (`examples/data/hr_requests.csv`): split without
  near-duplicate leaks, compare bare, prompt, plain and tempered combination and a cross-fitted cascade on accuracy,
  macro-F1 and worst-class recall.
- `examples/real_payloads/flagship_banking77.json` (the paper's Figure 4 requests and recorded answers, Banking77
  CC BY 4.0) and `examples/12_replay_flagship.py`: payload diffs and recorded probabilities offline, a byte-for-byte
  rebuild check, and a re-send with `--jev` when a key is set.
- `docs/why_it_works.md`: the fitted prior × evidence model, what it predicts and where it fails.
- `scipy` as an explicit dependency (it was already required by scikit-learn).

**Changed**

- README: Run 3 results and Run 4 slots in the results table, with the paper's corrected figures (the Run 3
  dose-response as cumulative costs 2.9 / 11.4 / 38.2 with steps −2.9 / −8.5 / −26.8; at 5 per class the prompt
  +2.1 [+0.0, +4.1] over bare Jev, +12.0 being the margin over the nearest centroid only); non-inferiority reported
  only as "no more than X points below"; R1 narrowed to pre-defined benign wrappers; the Jev 1.14 check.
- Decision guide rewritten to match the paper's Figure 7: one branch at about 100 labels per class (compare on your
  own held-out data), the cascade sends forwarded items with their nearest labelled examples, class shares only if
  live traffic matches them, and the small-class recall caveat.
- README: dollars per 1,000 decisions (cost is not the deciding factor), a table of what each approach sends to the
  vendor, what the on-disk cache stores, and how to move to a new model version.
- `docs/what_not_to_do.md` and `docs/algorithms.md`: Run 3 results, the precision target, cross-fitting, the tempered
  product and exponent fitting.
- Example docstrings reworded to the same evidence language; `examples/06` shows the cross-fitted report and the
  precision target, `examples/05` the tempered product.

## 0.1.0 (2026-11-09)

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
