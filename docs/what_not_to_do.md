# What not to do

The negative results behind the DON'T rules, with the paper's numbers. All results are for one model version
(Jev 1.13). Evidence levels:

- **Confirmatory**: a pre-registered test in Run 1 (held-out test items) or Run 2 (items never sent before), frozen
  by a hash before any call, Holm-corrected.
- **Exploratory**: development screens or analyses after the test labels were unsealed. Treat these as hypotheses
  to check on your own data.

Accuracy differences are in percentage points; brackets are 90% intervals.

---

## 1. Tuning the wording

Persona, ALL-CAPS emphasis and politeness add no information, and they did nothing.

- Each of the three prefixes stayed within ±2 points of the bare request on two datasets; all six intervals were
  inside [−1.3, +0.5]. **Confirmatory** (Run 1). The decision holds for accuracy averaged over two instruction
  wordings with item-level clusters; under a stricter class-cluster bootstrap with each wording separate it was
  inconclusive, and politeness cost about a point under one wording.
- Five further independently written personas and five politeness prefixes: −0.43 [−1.15, +0.28] and
  −0.47 [−1.02, +0.07]; no single stimulus moved accuracy by more than 1.2 points. **Exploratory.**
- A naive screening rule promoted 12 of 53 content-free edits. This is why a "+0.5 points" from a wording change
  on a few hundred items means nothing. **Exploratory** (measured).

Use `levers.add_persona` / `add_emphasis` / `add_politeness` only to reproduce the null on your data.

## 2. Wrong labels in the pool

Wrong labels remove information, and they hurt in proportion to the correct evidence they remove.

- Wrong labels on the three nearest shown examples: Banking77 93.6 → 60.1 (−33.5 [−38.8, −28.5]); financial
  tweets 78.6 → 65.2 (−13.4 [−15.9, −11.0]). **Confirmatory** (Run 1).
- The same corruption on a nearest-neighbour vote over the shown examples was far worse (Banking77 −71.5,
  tweets −39.5). **Exploratory.**
- Harm scales with the gold-labelled evidence removed (about 21 points per displaced gold example on Banking77,
  10 on tweets). **Exploratory.** A separate effect of rank at matched evidence was **not confirmed**
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

Prefer `correct_shares` after the call (it beat stating the shares by 1.3 and 2.0 points, **exploratory**), with
shares from the deployment period.

## 5. Ids or jargon names without definitions

- Cryptic tool ids alone: 2.3% accuracy on MetaTool (chance 5%); with one-line definitions 84.4%, non-inferior to
  readable names with the same definitions (−0.5 [−1.5, +0.4]). **Confirmatory** (Run 1; the queries of this
  dataset were generated from the descriptions, a caveat checked with hand-paraphrased definitions, −0.6, exploratory).
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

- With a threshold chosen on accuracy and frozen in advance, the cascade stayed non-inferior in accuracy on
  ToxicChat while sending 8.2% of items to the model (−0.2 [−0.8, +0.4]). **Confirmatory** (Run 2, accuracy only).
- On the same items it lost toxic-class precision −7.3 [−13.8, −1.1] and average precision −0.13; recall at
  precision 0.9 fell from 0.48 to 0.08. **Exploratory.** A cross-fitted threshold sending 37% kept precision.
- Bare Jev was 94% accurate at 7% prevalence while missing 61% of toxic prompts; a constant "not toxic" would be
  93% accurate. **Exploratory.**

Fit the threshold on the metric you care about (`ClassifierFirstCascade.fit(..., min_recall=..., positive=...)`).

## 10. Prompt wording as a security boundary

- Against an adaptive attacker without fabricated evidence, a guard sentence still let 93% of attacks succeed
  within 100 queries (median 9 queries); spotlighting 47% (median 23). With fabricated in-ticket evidence:
  guard 97%, spotlighting 85%. **Exploratory.**
- Splitting untrusted text into its own field did not stop pure evasion. **Exploratory.**

Treat the state as untrusted input. Enforce permissions outside the model.
