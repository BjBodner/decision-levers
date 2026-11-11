# Why it works: prior and evidence

A short account of why information moves a decision model and wording does not, and of the descriptive model in
the paper's Appendix I.13. That model is **observed**: its form was fixed before fitting but not deposited with a
third party, and it is fitted to Jev's outputs, not derived from Jev's internals. Nothing here observes how any
model works inside.

## The idea in plain words

A decision model answers from what it already knows about the options (its *prior*) and from what the request
tells it (the *evidence*). A wrapper such as a persona or a politeness line changes neither, so it stays within ±2
points. A retrieved example under its label is evidence for that label. A one-line description tells the model
what an option means; it matters most when the name says little, which is why it added 23.8 points on an invented
catalogue and less on MetaTool's readable names (10.7). A wrong label does two things at once: it removes correct
evidence and adds evidence for a rival. That is why deleting examples cost 13.1 points less than relabelling them,
and why wrong labels that all point at one rival cost 13.2 more than spread ones (Run 4).

## The model

For each option *c*, Jev's answer to a request with shown examples is described by

```
logit_c = a · log p_bare(c) + β · log( ε + Σ_{shown examples j labelled c} exp(τ · (cos_j − 1)) )
```

- `p_bare(c)` is Jev's own answer to the bare request: the **prior**.
- The sum is the **evidence**: each shown example labelled *c* adds a weight that falls off with its distance from
  the query (`cos_j` is the bge-small similarity, an external stand-in for whatever similarity Jev uses).
- `ε` is a floor: an option with no shown example keeps its prior instead of dropping to zero.

Fitted to 40,340 responses from Runs 1 and 2 (Banking77, CLINC150, tweets, CFPB): **a = 0.37, β = 1.92, τ = 13.9,
ε = 0.0225**. Across leave-one-dataset-out fits, *a* ranged from 0.347 to 0.409.

## What it predicts

- **The pooled wrong-label curve.** Applied without refitting to Run 3 (fresh items of four other datasets, 0 to 3
  of three correct shown examples relabelled), it gave 91.6, 88.4, 80.9 and 48.2% accuracy against 92.2, 89.4,
  80.9 and 54.1% observed. The floor `ε` produces the jump when the last correct example goes.
- **The direction of Run 4.** It got all three directions right and the deletion jump (+11.7 modelled, +10.1
  observed), but only about half of the two rival effects: delete minus relabel +7.0 modelled against +13.1
  observed, one rival minus distinct rivals −6.7 against −13.2. Jev weighs a rival's growing evidence more steeply
  than the model allows.

## Where it fails

- **Per dataset.** The pooled fit hides errors of 14 to 30 points in both directions on three of Run 3's four
  datasets.
- **Planted near-copies.** It underpredicted copying on Banking77-20 (83.5% predicted against 73.8% observed after
  one plant) and overpredicted it on tweets (15.5% against 49.5%), probably because it uses an external embedding.
- **Placement.** It has no placement term. On Jev the one-rival penalty was larger with the examples inside the
  option descriptions (13.2) than in a separate block (6.8); a block still showed a penalty, so matching the query
  against description text alone does not explain it. Rival models that only match descriptions fit worse
  (top-option agreement .70 to .73 against .91).

## Beyond Jev

- **Other decision models.** The harms carry over best: confusable pool noise and the one-rival penalty hurt on all
  six other models tested on fresh items, and on a chat model (the one-rival penalty on 5 of 6 on private data).
  That suggests, but does not show, that they are properties of evidence in the prompt rather than of typed decision
  APIs. The model above was fitted on Jev only, and the sizes differ by model: the one-rival penalty was smaller on
  every other model than on Jev, though scaled by each model's clean gain, Jev's loss can't be told apart from the
  two Perplexity models'.
- **Models that do not read examples as evidence.** On long private texts, Decider Flash steered away from the
  option its examples supported, and d1-ultra piled its answers onto one option. Neither fits a prior-plus-evidence
  account. Test examples on your model before you rely on them.
- **Leakage.** On four private synthetic datasets, eight models ranked eight techniques much as on public data (rank
  correlations 0.76 to 1.00; 0.49 to 0.94 on the six other models without the two largest effects), so leakage
  doesn't seem to drive the results. The paper's planned distortion test still flagged two techniques. Descriptions
  helped more on private data (+7.6 against +1.0 on the six others), which leakage would predict, but the names
  experiment points to a simpler cause: on OpsDeck half of each tool name was an invented brand word, so the
  description carried the meaning. Leakage isn't ruled out. The one-rival penalty was smaller on private data on
  three models, the opposite of what leakage would predict.

## What it does not license

- It is not a mechanism. It is a four-parameter curve that reproduces one pooled shape out of sample.
- The local combination in this package (`combine(bare, vote, lr)`) has the same prior × evidence shape, but it is
  not what Jev does inside, and it was not shown to match the examples prompt (−0.7 in both fresh-item tests). The
  tempered exponents (`PAPER_EXPONENTS`) are a fitted stacker, not estimates of `a` and `β`.
- It was fitted and tested on one model version (Jev 1.13).

Source: the paper's Appendices I.13 and I.14, Sections 4, 6 and 7.
