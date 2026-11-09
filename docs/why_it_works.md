# Why it works: a fitted prior × evidence model

A short account of the descriptive model in Section 6 of the paper: what it is, what it predicts, and where it
fails. It is **exploratory**: the form and its variants were written down before fitting, but not deposited with a
third party, and the model is fitted to Jev's outputs, not derived from Jev's internals. Nothing here observes how
Jev works inside.

## The model

For each option *c*, Jev's answer to a request with shown examples is described by

```
logit_c = a · log p_bare(c) + β · log( ε + Σ_{shown examples j labelled c} exp(τ · (cos_j − 1)) )
```

- `p_bare(c)` is Jev's own answer to the bare request (no examples): the **prior**.
- The sum is the **evidence**: each shown example labelled *c* adds a weight that falls off sharply with its
  distance from the query (`cos_j` is the cosine similarity of bge-small embeddings, an external stand-in for
  whatever similarity Jev uses).
- `ε` is a smoothing floor. An option with no shown example gets the same evidence term `β · log ε` as every other
  such option, so it keeps its relative prior instead of dropping to zero.

Fitted to Jev's kNN-20 probability vectors on Run 1 and Run 2 items of Banking77, CLINC150, financial tweets and
CFPB (40,340 requests): **a = 0.37, β = 1.92, τ = 13.9, ε = 0.023**. The prior weight stayed between 0.35 and 0.41
across leave-one-dataset-out fits. With τ ≈ 14, an example at cosine 0.8 weighs about 6% of an identical one.

Read plainly: Jev behaves as if it keeps about a third of its own prior and adds log-evidence from the nearest
labelled examples. That is why information levers move it (examples fill the evidence term; definitions and class
shares change the prior) and wording that carries no information does not.

## What it predicts

Applied without refitting to Run 3's dose-response items (fresh items of four other datasets, where 0 to 3 of three
correctly labelled shown examples were relabelled):

| Correct examples relabelled | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Observed accuracy, pooled | 92.2 | 89.4 | 80.9 | 54.1 |
| Predicted | 91.6 | 88.4 | 80.9 | 48.2 |

- **Within 1 point for 0 to 2 relabels**; the loss at 3 is overstated by 5.9 points [2.0, 9.9].
- **The convex shape.** The observed losses grow (cumulative 2.9, 11.4, 38.2 points; steps −2.9, −8.5, −26.8), and
  so do the steps in correct-option log-probability (observed −0.14, −0.32, −1.02 nats; predicted −0.12, −0.24,
  −1.16). The floor `ε` produces the jump: relabelling the last correct example drops its evidence term to `log ε`.
- **Only this form does it.** At 3 relabels a linear evidence term predicts 66.8%, a count without similarity
  weights 61.7%, and a pure vote without the prior 0.0%.
- On held-out datasets it fits Jev better than a pure vote or the bare prior alone on 3 of 4 (the vote misses the
  prior and agrees with Jev on only 57 to 66% of topical items).

## Where it fails

- **Per dataset.** It fails on 3 of 4 Run 3 datasets: at 3 relabels, predicted minus observed accuracy is −13.7
  points on LEDGAR, +22.9 on MetaTool and −30.5 on CFPB. It describes the pooled curve, not the spread between
  datasets; one global prior weight cannot fit sets where Jev leans on its prior more (CFPB, LEDGAR) or less
  (MetaTool).
- **Wrong labels on the nearest examples (tweets).** It underpredicts the harm: −4.6 points against −13.4 observed.
- **Planted near-copies.** It misses in both directions: after one wrong-labelled plant, 83.5% predicted against
  73.8% observed on Banking77-20, and 15.5% against 49.5% on tweets. The external embedding is the likely culprit.
- **Rival evidence (Run 4).** The paper's fourth run separated the two accounts of the growing loss, losing the
  last correct example and evidence piling up on rivals, and confirmed both (deletion costs less than relabelling,
  X3a; one common rival costs more than distinct rivals, X3b; deletion alone jumps at the last correct example,
  X3c). Applied without refitting to those 1,057 items, the model predicted the sign of every contrast, pooled and
  on each dataset, and the size of the deletion jump, but only about half of the two rival effects: with three
  correct examples deleted, 69.9% predicted against 75.9% observed; with two relabelled to one rival, 84.3% against
  74.2%. In the model's terms, Jev weighs a rival's accumulating evidence more steeply than the concave log-sum
  allows.

## What it does not license

- It is not a mechanism. It is a four-parameter curve that reproduces one pooled shape out of sample.
- It does not say the local combination in this package (`combine(bare, vote, lr)`) is what Jev does internally.
  That product has the same prior × evidence shape and was no more than 1.5 points below the retrieval prompt on
  tweets and complaints (non-inferior, Run 2), but it trailed on Banking77, CLINC150 and ToxicChat (exploratory),
  and it was not shown non-inferior on MASSIVE, LEDGAR and MetaTool (Run 4).
  The tempered exponents (`PAPER_EXPONENTS`) are a fitted stacker, not estimates of `a` and `β`: the bare
  exponent was lowest on Banking77 (0.18) and 0.32 to 0.54 elsewhere, with no intent-versus-topical split.
- It was fitted and tested on one model version (Jev 1.13).

Source: paper Section 6, Figure 6 and Table 4; analysis notes `round13/free_analyses.md` (A1, A2); Run 4 prediction
in `r14_x3_mechanism_prediction.csv` of the research package.
