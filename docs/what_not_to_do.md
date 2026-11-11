# What not to do

The negative results behind the don'ts, with the paper's numbers. Results are for Jev 1.13 unless a model is named.
Accuracy differences are in percentage points; brackets are 90% intervals. *Fresh* items are ones the study had
never sent to any model before that run. What the status words mean, as in the paper:

- **Confirmed**: a test planned before the run passed, under the stricter statistics (whole gold classes
  resampled).
- **Observed**: found after the fact, or passed only under looser statistics (items resampled).
- **Secondary**: planned as an estimate, with no pass or fail rule.
- **Not shown**: a planned test didn't pass. That never means "no effect".

"Within ±2" means two one-sided tests passed at ±2 points; it doesn't mean zero.

---

## 1. Expecting benign wrappers to help (Don't 1)

A *wrapper* is fixed text, the same on every item, that changes neither the correct option nor the rule for
picking it: a persona, a politeness prefix, a line in capitals, or unrelated questions in the same request.

- Five personas and five politeness prefixes, three instruction wordings, fresh items of four datasets: persona
  −0.3 [−1.0, +0.3], politeness −0.1 [−0.5, +0.4], both within ±2. **Confirmed** (Run 3).
- On fresh items, a capitals line was within ±2 of the bare call (+0.25), and so were seven unrelated questions
  sent along with ours (+0.30) (Run 8).
- Against a neutral prefix of the same length, persona and politeness scored +1.08 [−0.02, +2.22]: ±2 was not
  shown, so a gain of about a point is possible.
- Other models: within ±2 on 4 of 7 (Run 5). Politeness cost OpenAI Decisions 1.4 points.

This covers benign wrappers only. Format, option order, the option set, question polarity and rule wording weren't
varied, and they can move a model a lot: rewording a ticket-priority rule moved Jev by 24 points. Use
`levers.add_persona` / `add_emphasis` / `add_politeness` only to reproduce the null on your data.

## 2. Unchecked labels in the pool (Do 3, Don't 3)

A wrong label does two things: it removes correct evidence, and it adds evidence for a rival.

- **Pool noise, fresh items.** Relabelling 30% of the pool cost 3.9 points with uniform wrong labels and 7.0 with
  labels from the three nearest classes (*confusable* noise); at 50%, 10.2 and 19.3. With 30% confusable noise the
  examples prompt was no longer shown to beat the bare call (+0.7 [−1.5, +2.9]). **Confirmed** (Run 8).
- **Across models.** 30% confusable noise changed accuracy by −12.6 points [−20.9, −4.3] on average over six other
  models, with harm on 6 of 6. Four of them fell well below their own bare call (by 7.5 to 27.9 points), so for
  them a noisy pool is worse than no pool. **Confirmed** per model; the mean is observed (Run 8).
- **Two sources, within items** (1,057 fresh items showing three correct examples). Deleting the examples cost 13.1
  points less than relabelling them; pointing every wrong label at one rival cost 13.2 more than spreading them
  (the *one-rival penalty*; 13.8 on fresh items in Run 8); deleting the last correct example cost 10.1 more than
  deleting the second. **Confirmed** (Run 4).
- **The one-rival penalty across models**: −2.8 to −10.1 points, on 6 of 6 others on fresh items (**confirmed**,
  Run 8), and on 5 of 6 on private data, where Kev-4B reversed (Run 10). It was smaller on every other model than
  on Jev, though scaled by each model's clean gain, Jev's loss can't be told apart from the two Perplexity models'.
- **The three-example test.** On easy items, relabelling one, two or three of three correct examples cost 2.9,
  11.4 and 38.2 points (slope confirmed, Run 3); on other item sets the full relabel cost 49.5 and 60.6. This test
  shows the shape of the loss; the pool-noise costs are the sizes a team should expect.
- **Auditing helps a little.** Checking a random 10% of a noisy pool won back 0.7 points, the 10% most often
  retrieved 1.0; the second was not shown to beat the first. **Secondary.**

A missing example costs less than a wrong one: drop a doubtful example rather than keep its label.
`LabelledPool.with_wrong_labels` (uniform noise) lets you measure your own exposure.

## 3. Assuming placement is neutral (Don't 4)

- On Jev, the one-rival penalty was larger with the examples inside the option descriptions than in one labelled
  block after the instruction: 13.2 against 6.8 points (difference +6.4). With equal-length descriptions it stayed
  at 7.4, with each label named once in the block 8.1, and on fresh items 6.4. **Confirmed** (Runs 6 to 8). Clean
  examples helped about equally in both places.
- Other models differ: larger inside descriptions on Decider Flash (+9.0), Microsoft-Decision-1 (+7.5), Kev-4B
  (+3.9) and d1-ultra (+3.2); smaller on Decider (−15.9), OpenAI Decisions (−4.0) and the chat model (−4.8). So
  placement depends on the model.
- Placement wasn't tested with a noisy pool, so a block for a noisy pool is only a hypothesis.

`add_examples` puts the examples in the option descriptions, as the paper's main arms did. If wrong labels may pile
on one rival, test the placement on your model.

## 4. Planted near-duplicates (pool poisoning)

- One, two or three planted near-copies of each item with one wrong label: Banking77-20 fell from 92.2% to 74.0%,
  39.3% and 23.5%; tweets from 74.5% to 49.5%, 32.5% and 25.8%. **Observed** (development items).
- De-duplicating the pool changed nothing at one or two plants: synonym-swapped copies survive.

If outsiders can write to your pool, retrieval in the prompt is an attack surface.

## 5. Unmeasured class shares (Don't 2)

- Stating last year's shares for 2026 complaints cost 11.5 points against the true shares. **Observed** (Run 7; with
  8 classes the class-level test didn't pass). All six other models lost accuracy on the central estimate (−3.9
  to −16.3), but only two significantly, and none after one correction across models.
- Shuffled shares cost 6.2 to 6.8 points against true shares (Run 1) and 1.1 against none (LEDGAR, Run 8).
  **Observed.**
- True shares were not shown to help: +1.9 (Run 2, observed) and −0.4 (Run 8, not shown). On development
  ToxicChat they cut toxic recall from 39.4% to 33.0%.
- Shares estimated from 100 labelled recent items came within about a point of the true shares.

If you use shares, measure them first. `correct_shares` applies them to the bare answer after the call; that beat
stating them by 1.3 and 2.0 points on development data (observed).

## 6. Names that say nothing (Do 2)

- On MetaTool, one-word tool names cost 10.7 points against two-word names. On an invented catalogue (OpsDeck-24),
  the function word alone (*Queue*) did about as well as the full name, but the brand word alone (*Mossgate*) fell
  to 5.0% accuracy, close to the 2.1% of cryptic ids. **Secondary** (Run 9).
- A one-line description added 10.7 points on MetaTool and 23.8 on the invented catalogue (16.8 to 39.8 on six
  other models). Two caveats: MetaTool's queries were generated from its own descriptions, and where names were
  already clear, descriptions did little (−1.0 and −0.4 on CLINC150 and LEDGAR).
- With a description on every option, ids came within a point of names (−0.8 and 0.0). But in a planned test on
  fresh items, ids with definitions were 2.3 points below names and didn't pass the test that they are no more
  than 2 points worse; on other models it passed on only 1 of 6 (Run 8). Readable names stay the default.

## 7. Naive pool cleaning with the model

- Dropping pool items whose label the bare call disputes did not help: Banking77-20 80.0% against 80.2% for the
  noisy pool, tweets 73.5% against 74.1%. **Observed.**

## 8. Retrieval without checking your task and model

- On development data, long-text CFPB and LEDGAR gained only +1.0 to +1.4 from retrieval, and knowledge-bound tasks
  and gates moved by at most 1.0 point. **Observed.**
- **Long texts on other models.** On a private set of long lease clauses, Decider Flash fell from 98.9% to 28.1%
  with three correctly labelled examples, yet scored 96.9% when they carried wrong labels: it steers away from the
  option the examples support. It fell on a private ticket set too (94.7% to 52.7%), and on public LEDGAR (39 to
  45% against 93% bare). d1-ultra fell on the ticket set from 90.6% to 53.2% with nearest examples, and its
  examples didn't help on public LEDGAR either, so this is unlikely to be leakage. **Observed** (Run 10).

Test examples on your model, on your longest texts, before you rely on them.

## 9. Judging rare classes by accuracy (Don't 5)

- A classifier-first cascade with an accuracy-chosen threshold sent 8.2% of ToxicChat items to the model and was no
  more than 1.5 points worse on accuracy (−0.2; **confirmed** for these two classes), but toxic-class precision fell
  7.3 points [1.1, 13.8] and recall at precision 0.9 fell from 0.48 to 0.08 (**observed**). On Run 1 items a
  threshold sending 37% kept precision (−0.3 [−4.0, +1.8]).
- The local combination at 100 per class had a mean worst-class recall of 29.7% against 42.8% for the prompt
  (secondary). On a CFPB time split (2025 pool, 2026 complaints) it fell 10.9 points below the prompt, with 17.3%
  recall on payday loans.

Fit the cascade threshold on the metric you care about (`ClassifierFirstCascade.fit(..., min_recall=...,
min_precision=..., positive=...)`), quote the cross-fitted report (`cv=5`), and report per-class recall
(`evaluate.class_metrics`) next to accuracy.

## 10. Prompt wording as a security boundary (Don't 6)

- A one-line claim that a wrong team was correct cut ticket-routing accuracy from 87% to 33%.
- A guard sentence, spotlighting and a separate claim field all fell to a search over 100 attack variants that
  never saw the model's scores (success with the true label kept: at least 38%, 15% and 55%). **Observed.**

Content, not wording, moves these models. Treat the text as untrusted input, and enforce permissions outside the
model.

## 11. Leakage as an excuse

Public benchmarks may be in a vendor's training data, so the paper re-ran eight techniques on four private
synthetic datasets with eight models (Runs 9 and 10). The techniques ranked much the same on public and private
data (rank correlations 0.76 to 1.00; 0.49 to 0.94 on the six other models without the two largest effects), and
cryptic ids, three wrong labels and pool noise hurt on private data on every model. So leakage doesn't seem to
drive the harms above, though the paper's planned distortion test flagged two techniques: descriptions helped more
on private data, and the one-rival penalty was smaller there on three models (and reversed on Kev-4B). The private
items were written by language models and may be cleaner than real ones.
