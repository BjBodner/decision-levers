# What not to do

The negative results behind the DON'T rules, with the paper's numbers. All results are for Jev 1.13 (a small
Jev 1.14 check is summarised in the README). Evidence levels:

- **Confirmatory**: a pre-registered test in Run 1 (held-out test items), Run 2 (items never sent before) or Run 3
  (fresh items of four further datasets, deposited publicly before the first call, gold-class clusters), frozen
  by a hash before any call, Holm-corrected. A non-inferiority result is written "no more than X points below".
- **Exploratory**: development screens or analyses after the test labels were unsealed. Treat these as hypotheses
  to check on your own data.

Accuracy differences are in percentage points; brackets are 90% intervals.

---

## 1. Tuning the wording

Pre-defined benign wrappers (a persona, politeness, capitals, unrelated questions packed alongside) add no
information about the label, and they stayed within ±2 accuracy points. This covers only such wrappers: reworded
decision rules, renamed options, natural answer-preserving context and adversarial suffixes do move Jev (studies
cited in the paper, Section 3). Test those like any content change.

- Each of the three prefixes stayed within ±2 points of the bare request on two datasets; all six intervals were
  inside [−1.3, +0.5]. **Confirmatory** (Run 1, accuracy averaged over two instruction wordings, item-level
  clusters). Politeness cost about a point under one wording.
- Five independently written personas and five politeness prefixes, each under three paraphrases, on fresh items
  of four datasets with stimuli, paraphrases and gold classes resampled: persona −0.3 [−1.0, +0.3], politeness
  −0.1 [−0.5, +0.4], both within ±2. **Confirmatory** (Run 3). Post hoc, no single stimulus moved a dataset by more
  than 1.83 points with paraphrases pooled, but single stimulus × paraphrase cells of 200 items reached −4.0
  [−7.1, −1.0] (4 of 120 cells beyond ±2, no multiplicity control). **Exploratory.**
- A naive screening rule promoted 12 of 53 content-free edits. This is why a "+0.5 points" from a wording change
  on a few hundred items means nothing. **Exploratory** (measured).

Within ±2 is not zero. Use `levers.add_persona` / `add_emphasis` / `add_politeness` only to reproduce the null on
your data.

## 2. Wrong labels in the pool

Wrong labels remove information, and the harm grows with each correct example relabelled.

- Wrong labels on the three nearest shown examples: Banking77 93.6 → 60.1 (−33.5 [−38.8, −28.5]); financial
  tweets 78.6 → 65.2 (−13.4 [−15.9, −11.0]). **Confirmatory** (Run 1).
- A planted dose on fresh items of four datasets, relabelling 0 to 3 of three correctly labelled shown examples:
  slope −12.3 [−13.3, −11.3] points per example, every step negative. **Confirmatory** (Run 3; "monotone").
  Descriptively, relabelling one, two and three of three cost 2.9, 11.4 and 38.2 points (steps −2.9, −8.5,
  −26.8): the loss grows, so "linear" or "in proportion" is not supported. Most of the harm comes when the last
  correct example goes.
- The same corruption on a nearest-neighbour vote over the shown examples was far worse (Banking77 −71.5,
  tweets −39.5). **Exploratory.**
- Observationally, about 21 points per displaced gold example on Banking77 and 10 on tweets. **Exploratory.** A separate effect of rank at matched evidence was **not confirmed**
  (+1.2 [−0.4, +2.7], Run 2).
- 30% wrong pool labels gave back more than half of the retrieval gain in toxic-class recall on ToxicChat
  (62% → 49%); 10% was harmless there. **Exploratory.**

Audit the labels of the examples most likely to be retrieved.

## 3. Planted near-duplicates (pool poisoning)

- One planted near-copy of each item with a wrong label: Banking77 (20 intents) 92.2 → 74.0, tweets 74.5 → 49.5;
  two copies 39.3 and 32.5. **Exploratory** (development items).
- The prompt was MORE vulnerable to a single planted twin than a 5-nearest-neighbour vote. **Exploratory.**
- De-duplicating the pool changed nothing (synonym-swapped copies survive). Showing one example per label with a
  two-label agreement rule cut the one-twin damage but cost clean accuracy and failed from two twins.
  **Exploratory.**

If outsiders can write to your pool, retrieval in the prompt is an attack surface.

## 4. Shuffled, uniform or stale class shares

- True shares beat the most distant of 200 sampled share permutations by +6.8 (tweets) and +6.2 (CFPB).
  **Confirmatory** (Run 1). Most of that gap is the cost of the wrong shares: true shares beat no shares by only
  +1.9 [+1.1, +2.7]. **Confirmatory** (Run 2; mostly tweets, and not significant under class clustering).
- Stating uniform shares was worse than stating none on tweets (−1.5 [−2.3, −0.7]). **Exploratory.**
- Stating training-period shares cost 5 to 8 points when the class mix had shifted. **Exploratory.**
- Even true shares push answers to frequent classes: on tweets accuracy rose 70.9 → 71.9 while macro-F1 fell
  .661 → .608. **Exploratory.**

Use shares only if live traffic matches them. Otherwise re-estimate them and prefer `correct_shares` after the call
(it beat stating the shares by 1.3 and 2.0 points, **exploratory**). Both routes trade minority recall for accuracy.

## 5. Ids or jargon names without definitions

- Cryptic tool ids alone: 2.3% accuracy on MetaTool (chance 5%); with one-line definitions 84.4%, no more than 2
  points below readable names with the same definitions (−0.5 [−1.5, +0.4], non-inferior). **Confirmatory** (Run 1; the queries of this
  dataset were generated from the descriptions, a caveat checked with hand-paraphrased definitions, −0.6, exploratory).
- One-line definitions written blind from the label names, on three datasets whose queries were not generated from
  them (MASSIVE, LEDGAR, CFPB): +73.2 [+63.2, +83.5] points over cryptic ids, and no more than 2 points below names
  with the same definitions on average (−1.1 [−1.8, −0.5]), but not consistently: LEDGAR was 2.4 below.
  **Confirmatory** (Run 3).
- Neutral ids on three other datasets: 3.0%, 2.5% and 16.3%. **Exploratory.**
- Stating class shares over ids the model cannot read reached only 14.5% and was flagged as harmful on gold
  log-probability, against 87.1% with definitions. **Exploratory.**

## 6. Naive pool cleaning with the model

- Dropping pool items whose label the bare model disagrees with (threshold 0.7) HURT: Banking77 (20 intents)
  87.0 → 80.0, tweets 75.7 → 73.5, worse than keeping a 30%-noisy pool. The model disagrees with many correct
  hard labels. **Exploratory.**
- See also 3: de-duplication is not a poisoning defense.

## 7. A labels-only shortlist on topical text

- On financial tweets the shortlist scored 69.3% against 70.9% for the bare model (and 77.1% for the kNN-20
  prompt); the gold label was in the shortlist only 77.8% of the time. **Exploratory.**
- It works on short, balanced intent sets (Banking77: 91.4% vs 93.7% for the kNN-20 prompt at 62 vs 1,117 tokens).
  **Exploratory.**

## 8. Retrieval where the decision is knowledge-bound

- Retrieved examples did nothing where bare accuracy was already high or the decision depends on world knowledge
  rather than on labelling conventions: When2Call 87.6 → 88.6, a function-calling gate 77.7 → 77.9, FOMC
  hawkish/dovish 67.9 → 68.5, PubMedQA 75.2 → 75.2. **Exploratory** (development items).
- It helped a lot where labels are specific to the dataset (MetaTool +11.5, CVSS severity +17.2).
  **Exploratory.**

Retrieval costs tokens and sends pool text; check that your task has headroom first.

## 9. Accuracy-only cascade thresholds on rare positives

- With a threshold chosen on accuracy and frozen in advance, the cascade sent 8.2% of ToxicChat items to the model
  and was no more than 1.5 points below the prompt on accuracy (−0.2 [−0.8, +0.4]). **Confirmatory** (Run 2,
  accuracy only).
- On the same items it lost toxic-class precision −7.3 [−13.8, −1.1] and average precision −0.13; recall at
  precision 0.9 fell from 0.48 to 0.08. **Exploratory.** A cross-fitted threshold sending 37% kept precision.
- Bare Jev was 94% accurate at 7% prevalence while missing 61% of toxic prompts; a constant "not toxic" would be
  93% accurate. **Exploratory.**

Fit the threshold on the metric you care about (`ClassifierFirstCascade.fit(..., min_recall=..., min_precision=...,
positive=...)`), and quote the cross-fitted report (`cv=5`): a threshold chosen on the same items it is scored on
is optimistic. The forwarded items are sent with their nearest labelled pool texts.

## 10. Prompt wording as a security boundary

- Against an adaptive attacker without fabricated evidence, a guard sentence still let 93% of attacks succeed
  within 100 queries (median 9 queries); spotlighting 47% (median 23). With fabricated in-ticket evidence:
  guard 97%, spotlighting 85%. **Exploratory.**
- Splitting untrusted text into its own field did not stop pure evasion. **Exploratory.**

Treat the state as untrusted input. Enforce permissions outside the model.

## 11. Judging the local combination by accuracy alone

- At about 100 labels per class, prompt minus local combination (`combine(bare, vote, lr)`) on macro-F1 ranged
  from −3.1 to +1.0 points across Runs 2 and 3 (post hoc), but on CFPB the combination's recall on the smallest class (Payday loan) was 10 to 16
  points below the prompt's lowest class recall: 58.9 against 68.7 (Run 2 items) and 52.6 against 68.8 (Run 3
  items); intervals cross 0. **Exploratory.**
- Any expert in a product can veto a class: a near-zero vote or classifier probability outweighs a right bare call.

Report per-class recall (`evaluate.class_metrics`) next to accuracy before swapping the prompt for the combination.
