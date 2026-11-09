# decision-levers

**Information, not instructions.** Tested prompting levers for typed decision models, and the ones that do nothing.

> Independent project. Not affiliated with, funded by, or endorsed by TypeSafe. 'Jev' and 'TypeSafe' are used only to name the API this code calls.

Paper: arXiv link added on publication.

A typed decision model takes a text, a closed list of options and an instruction, and returns a probability for
each option. It answers from two sources: what it already knows (its prior), and what the request tells it (the
evidence). Pre-defined benign wrappers, which carry no information about the label (a persona, politeness,
capitals, unrelated questions packed alongside), stayed within ±2 accuracy points. Edits that add information
(retrieved labelled examples, one-line definitions, true class shares) helped, and wrong information (mislabelled
examples, shuffled shares) hurt. With enough labels, prior and evidence can also be combined outside the prompt, by
multiplying one bare call by local models; whether that is good enough for your data is something to measure on
your data. This package implements each approach the paper tested on TypeSafe Jev 1.13, plus the evaluation
protocol that tells real effects from noise. Results come from four pre-registered runs (25 of 29 tests passed),
the third deposited publicly before its first call.

## Which approach?

The label budget decides where to combine the model's prior and the evidence. **C1/C2/C3/C4** = confirmed in Run
1/2/3/4; **E** = exploratory; "item-level" = not confirmed under class clustering.

**How many labelled examples per class do you have?**

- **About 5.** Put the nearest labelled examples in the prompt (`add_examples`). Run 2: +3.6 and +2.1 points over
  the bare call and +17.3 and +16.1 over a tuned classifier on tweets and complaints (C2, item-level). Run 3, four
  more datasets: +2.1 [+0.0, +4.1] over the bare call and +10.8 over a tuned classifier (secondary, no decision).
  Run 4, three more datasets: +2.9 [+0.7, +4.9] over the bare call and +9.1 [+7.3, +11.2] over a tuned classifier
  (secondary). A bare call multiplied by a local classifier was not shown non-inferior to the prompt on those three
  datasets (Run 4: X1a), so its standing at this budget is not established.
- **About 100.** Compare one bare call × a kNN vote × a tuned classifier (`combine(bare, vote, lr)`, run on your
  machine) with the prompt **on your own held-out data** (`examples/11`). It was no more than 1.5 points below the
  prompt on tweets and complaints (non-inferior; C2, item-level); on fresh items of four further datasets it was
  inconclusive (Run 3), and on three further datasets it was not shown non-inferior (Run 4: X1b). Across six pool
  draws of the Run 2 items the difference was above −1.5 every time but on both sides of zero (Run 4, secondary),
  so one pool draw is not enough to decide. It sends only the item. **Small-class caveat:** on CFPB its recall on
  the smallest class was 10 to 16 points below the prompt's lowest class recall (E; intervals cross 0), so compare
  per-class recall, not just accuracy (`evaluate.class_metrics`).
- **Time-split caution.** If your labelled pool is older than the traffic you will score, and holds the natural
  class mix (a handful of examples for the tail classes), check both approaches on recent held-out items before
  choosing. On CFPB, with a 2025 pool of 800 complaints at natural shares and 2,000 complaints received in 2026,
  the retrieval prompt gave no gain over the bare call (+0.05 [−0.6, +0.8]), and the local combination fell 10.9
  points below the prompt, with 17.3% recall on payday loans (7 pool examples; Run 4, secondary). The class mix had
  shifted too: credit reporting was 75.8% of the 2025 complaints and 3.0% of the 2026 test items.
- **If vendor calls must be cut:** a classifier-first cascade (`ClassifierFirstCascade`). The local classifier
  answers the items it is sure of; **the other items go to the prompt with their nearest labelled examples**, so a
  cascade reduces what leaves your machine but does not remove pool text. On ToxicChat it sent 8% of items and was
  no more than 1.5 points below the prompt on accuracy (C2), but toxic-class precision fell 7.3 points (E): choose
  the threshold on recall or precision, cross-fitted.

**At any budget**

- **Always:** define cryptic option names (no more than 2 points below readable names on average, C1, C3; LEDGAR
  2.4 below). **Use class shares only if live traffic matches them** (+1.9 over none, C2, item-level); otherwise
  re-estimate them and correct the bare answer after the call (E). Both routes trade minority recall for accuracy
  and lost 5 to 8 points under a shifted class mix (E). **Confirm before you ship** (`examples/09`): held-out items
  de-duplicated against the pool, content-free controls, analysis frozen before testing. The paper's tests used
  938 to 3,424 items; a team with 5 labels per class may not have that many, and should read the 5-per-class rule
  as the paper's confirmation, not theirs (E).
- **Never:** expect a persona, politeness, capitals or packed unrelated questions to help (all within ±2 points;
  C1, C3). Trust pool labels unchecked (three wrong labels cost 13 to 34 points in Run 1; in Run 3 the harm grew
  with each correct example relabelled). A wrong label both removes correct evidence and adds evidence for a rival:
  it costs most when it takes the last correct example, and more when several wrong labels point to one class
  (C4). Judge rare classes by accuracy (use precision and recall at your operating point).
- **Further cautions (E):** prompt wording, spotlighting and a separate claim field are not security boundaries,
  and screen results need a magnitude floor. Vendor cost ($0.0008 to $0.13 per 1,000 decisions) will not decide.

R1, the "within ±2" result, covers pre-defined benign wrappers only. Reworded decision rules, renamed options,
natural answer-preserving context and adversarial suffixes do move Jev (cited in the paper); test them as content.

| Your situation | Try first | Example |
|---|---|---|
| No labels | Bare request; one-line definitions if option names are ids or jargon; shares only if live traffic matches, else post-hoc correction with re-estimated shares | `01`, `03`, `04` |
| About 5 labels per class | Retrieved examples in the prompt | `02` |
| About 100 labels per class | Compare the local combination (plain or tempered) with the prompt on your own held-out data, per-class recall included | `05`, `11` |
| Pool older than live traffic, natural class mix | Score both approaches on recent held-out items; check tail-class recall before using the local combination | `11` (its split is random: hold out your most recent rows yourself) |
| Vendor calls must be cut | Classifier-first cascade, threshold on recall or precision, cross-fitted | `06` |
| Short, balanced intent sets, cost-bound | Labels-only shortlist (most items need no call) | `07` |
| Several decisions about the same text | Pack them into one request, main question last | `08` |
| Before shipping anything | Held-out, de-duplicated items; null controls; freeze; confirm | `09` |
| Your own CSV | Load, split without near-duplicate leaks, compare every approach | `11` |
| See a real request and answer | The paper's flagship Banking77 payloads, replayed offline | `12` |

Details and evidence levels for each DON'T: [docs/what_not_to_do.md](docs/what_not_to_do.md). Pseudocode for
every approach and for the protocol: [docs/algorithms.md](docs/algorithms.md). The fitted prior × evidence model,
what it predicts and where it fails: [docs/why_it_works.md](docs/why_it_works.md).

## Quickstart

```bash
pip install -e ".[dev]"            # or: pip install -e ".[dense]" for dense retrieval (fastembed)
python examples/02_retrieved_examples.py           # offline mock backend, free
python examples/11_your_own_data.py --csv my.csv --text-col body --label-col queue
pytest
```

```python
from decision_levers import (ClassifierFirstCascade, LabelledPool, MockBackend, Request, TunedLR, add_examples,
                             combine, knn_vote, tempered_combine)

pool = LabelledPool.from_pairs([("I was charged twice", "Billing"), ("my parcel never came", "Shipping"),
                                ("invoice shows the wrong plan", "Billing"), ("tracking stopped updating", "Shipping")])
# or: LabelledPool.from_csv("tickets.csv", text_col="body", label_col="queue", where={"split": "train"})
req = Request(state="I was charged twice on this invoice", question="Which team should handle this ticket?",
              options=["Billing", "Shipping"])
backend = MockBackend()                 # JevBackend(PAPER_MODEL) for the real API

p_prompt = backend.decide(add_examples(req, pool))           # examples in the prompt (sends pool text)
bare = backend.decide(req)                                    # or: one bare call ...
vote = knn_vote(req.state, pool, req.options)
lr = TunedLR(pool, list(req.options)).fit().predict([req.state])[0]
p_local = combine(bare, vote, lr)                             # ... combined locally (plain product, G5)
p_temp = tempered_combine(bare, vote, lr)                     # tempered: the paper's frozen exponents (E)
```

The tempered product raises the three inputs to exponents, by default the paper's frozen set for about 100 labels
per class, `PAPER_EXPONENTS = (0.358, 0.131, 0.620)` (bare, vote, classifier; exploratory, fitted with the paper's
dense features). Refit them on your own held-out items with `fit_exponents(bare, [votes, lrs], gold, cv=5)` and keep
the plain product unless the cross-fitted log-loss is lower.

The cascade threshold is fitted on held-out items. A threshold chosen on the same items it is scored on is
optimistic, so ask for a cross-fitted estimate:

```python
cas = ClassifierFirstCascade().fit(local_answers, model_answers, gold, min_recall=0.6, min_precision=0.8,
                                   positive="toxic", cv=5, groups=near_dup_groups)
cas.report["crossfit"]        # out-of-fold share sent, accuracy, recall, precision: quote these
```

To use the real API, set `TYPESAFE_API_KEY` (or `JEV_API_KEY`) and pin the model:

```python
from decision_levers import JevBackend, PAPER_MODEL
backend = JevBackend(PAPER_MODEL)       # "jev-1.13.0"; a response from another version raises
```

Every example takes `--mock` (default, offline) or `--jev` (paid; responses are cached on disk, so reruns are free).
The mock is a keyword-overlap fake built to show the mechanics; it mimics the direction of the paper's findings by
construction, so its numbers are not evidence. Every mock run says so.

## Results from the paper

All on Jev 1.13. **Confirmatory** = pre-registered, frozen by a hash before any call, Holm-corrected: Run 1 used
held-out test items, Run 2 items never sent in any request, Run 3 fresh items of four further datasets (MASSIVE,
LEDGAR, MetaTool, new CFPB rows), deposited publicly before its first call and clustered by gold class, Run 4 fresh
items of MASSIVE, LEDGAR and MetaTool, frozen internally (not deposited) and clustered by gold class.
**Secondary** = pre-specified estimates without a decision. **Exploratory** = development screens or analyses after
unsealing. Differences in accuracy points, 90% intervals in brackets. A non-inferiority result is reported as "no
more than X points below", never as equal.

| Lever | Result | Evidence |
|---|---|---|
| **Wording (R1)** | | |
| Persona, capitals, politeness vs bare (tweets, CFPB) | each within ±2; six intervals inside [−1.3, +0.5] | Confirmatory, Run 1 |
| Seven unrelated questions packed with the main one | within ±2 | Confirmatory, Run 1 |
| 5 personas and 5 politeness prefixes × 3 paraphrases vs bare, 4 datasets | persona −0.3 [−1.0, +0.3], politeness −0.1 [−0.5, +0.4]; both within ±2 | Confirmatory, Run 3 |
| Same, finer cells (stimulus × paraphrase, 200 items each) | 4 of 120 cells beyond ±2, down to −4.0 [−7.1, −1.0] | Exploratory |
| **Information (R2)** | | |
| Retrieved examples (kNN-20, 80 to 100 labels per class) vs bare | +11.2 [+7.5, +15.2] Banking77; +5.9 [+4.2, +7.6] CLINC150; +8.7 [+7.5, +10.0] financial tweets | Confirmatory, Run 1 |
| True class shares vs none (tweets + CFPB) | +1.9 [+1.1, +2.7] (tweets +2.9, CFPB +0.9) | Confirmatory, Run 2 (item-level) |
| True vs shuffled class shares | +6.8 tweets, +6.2 CFPB | Confirmatory, Run 1 |
| Shares applied after the call vs stated | +1.3 tweets, +2.0 CFPB | Exploratory |
| Definitions on readable tool names (MetaTool) | +5.8 [+4.3, +7.1] | Confirmatory, Run 1 |
| Cryptic ids + definitions vs names + definitions (MetaTool) | −0.5 [−1.5, +0.4]: no more than 2 points below (ids alone: 2.3%) | Confirmatory, Run 1 |
| Ids + definitions written blind vs ids alone; vs names + definitions (MASSIVE, LEDGAR, CFPB) | +73.2 [+63.2, +83.5]; −1.1 [−1.8, −0.5]: no more than 2 points below on average, but LEDGAR 2.4 below | Confirmatory, Run 3 (not consistent across datasets) |
| Wrong labels on the 3 nearest shown examples | −33.5 Banking77, −13.4 tweets | Confirmatory, Run 1 |
| Relabelling 0 to 3 of three correct shown examples, 4 datasets | slope −12.3 [−13.3, −11.3] per example; cumulative cost 2.9, 11.4, 38.2 (steps −2.9, −8.5, −26.8; descriptive) | Confirmatory, Run 3 (monotone; not linear) |
| Rank effect at matched evidence (nearest vs random ranks relabelled) | +1.2 [−0.4, +2.7], not confirmed | Confirmatory, Run 2 |
| **Label budget (R3)** | | |
| 5 per class: kNN-20 prompt vs a tuned classifier / vs bare | tweets +17.3 / +3.6; CFPB +16.1 / +2.1 | Confirmatory, Run 2 (item-level) |
| 5 per class, 4 datasets: prompt vs bare; vs tuned classifier; vs nearest centroid | +2.1 [+0.0, +4.1]; +10.8 [+7.4, +15.0]; +12.0 [+8.1, +17.9] | Secondary, Run 3 |
| 5 per class: bare × nearest centroid vs the prompt | +2.1 [+1.1, +3.1] for the product (chosen post hoc) | Exploratory |
| 100 per class: one bare call × vote × tuned classifier vs the kNN-20 prompt (tweets + CFPB) | +1.4 [+0.3, +2.5]: no more than 1.5 points below the prompt | Confirmatory, Run 2 (item-level; class-cluster p .052) |
| Same, LEDGAR + new CFPB rows / MASSIVE + MetaTool | +0.1 [−2.1, +1.5] / +0.6 [−0.7, +2.0]: inconclusive | Secondary, Run 3 |
| Same, Banking77, CLINC150, ToxicChat | −0.8 to −1.5, trails the prompt | Exploratory |
| Tempered product (exponents cross-fitted on gold) vs the prompt | Banking77 +0.2 [−0.4, +0.7]; Run 2 ToxicChat −0.4 [−1.0, +0.1]; CLINC150 −0.9 [−1.4, −0.5] | Exploratory |
| Plain product, recall on CFPB's smallest class (Payday loan) vs the prompt's lowest class recall | 58.9 vs 68.7 (Run 2 items), 52.6 vs 68.8 (Run 3 items); intervals cross 0 | Exploratory |
| Retrieval prompt vs a fixed kNN vote (Banking77 + CLINC150) | +3.0 [+2.4, +3.7]: the prompt is better, not equivalent | Confirmatory, Run 1 (test not passed) |
| Classifier-first cascade, ToxicChat, 8.2% of items sent | −0.2 [−0.8, +0.4]: no more than 1.5 points below the prompt on accuracy | Confirmatory, Run 2 |
| Same cascade, toxic-class precision | −7.3 [−13.8, −1.1]; on Run 1 items a cross-fitted threshold sending 37% kept precision (−0.3) | Exploratory |
| Cross-fitted cascades, 5 datasets | 0.5 to 0.75 points below full kNN-20 at 1.7 to 31.6% of items sent | Exploratory |
| **Other** | | |
| Three related questions packed vs separate (CFPB) | Δ 0.0 on the main question; 51% fewer input tokens | Exploratory |
| Labels-only shortlist vs kNN-20 prompt (Banking77) | 91.4 vs 93.7; 62 vs 1,117 tokens; 80% of items without a call | Exploratory |
| Naive screening rule on content-free edits | promoted or flagged 12 of 53 | Exploratory |
| Jev 1.14 vs 1.13, four arms on CFPB and MetaTool | point estimates within 2.5 points; under class clusters retrieval and top-3 harm met the pre-registered direction on MetaTool, not CFPB; CFPB persona difference +1.0 [+0.1, +1.8] | Secondary, Run 3 |
| **Run 4** (fresh MASSIVE, LEDGAR, MetaTool; class clusters; Holm over its 5 tests) | | |
| Why wrong labels hurt: deleting k correct shown examples vs relabelling them to distinct rivals (mean over k = 1 to 3; 1,057 items) | +13.1 [+11.7, +14.7]: deletion costs less than relabelling | Confirmatory, Run 4 (X3a) |
| Relabelling to one common rival vs to distinct rivals (mean over k = 2, 3) | −13.2 [−14.7, −11.7]: one rival costs more | Confirmatory, Run 4 (X3b) |
| Deletion alone: loss at the third (last) correct example minus loss at the second | +10.1 [+7.5, +12.7]: the loss jumps at the last correct example (deletion steps −1.8, −3.8, −13.9; descriptive) | Confirmatory, Run 4 (X3c) |
| All three correct examples changed: deleted / relabelled to distinct rivals / to one rival, vs the clean display | −19.4 / −47.0 / −67.2 | Descriptive, Run 4 |
| 5 per class: one bare call × tuned classifier vs the kNN-20 prompt (5,606 items, 104 classes) | −0.6 [−1.7, +0.3]: not shown non-inferior at −1.5 (Holm p .16); below 0 on every dataset | Confirmatory, Run 4 (X1a; test not passed) |
| 100 per class: bare × vote × tuned classifier (the G5 product) vs the prompt | −0.7 [−1.6, +0.1]: not shown non-inferior at −1.5 (Holm p .16); below 0 on every dataset | Confirmatory, Run 4 (X1b; test not passed) |
| Same two tests with near-duplicate clusters instead of class clusters | X1a [−1.2, −0.1], X1b [−1.3, −0.2]: both intervals above −1.5 | Secondary, Run 4 (the pre-registered unit is the class) |
| 100 per class: tempered product (exponents frozen before Run 4) vs the prompt | −0.4 [−1.2, +0.3] | Secondary, Run 4 |
| 5 per class: prompt vs bare; vs tuned classifier; vs nearest centroid | +2.9 [+0.7, +4.9]; +9.1 [+7.3, +11.2]; +9.8 [+7.8, +12.1] | Secondary, Run 4 |
| 5 per class: bare × nearest centroid vs the prompt | −0.5 [−1.9, +1.2]: the post hoc +2.1 did not recur on fresh items | Secondary, Run 4 |
| SuperICL (the classifier's label and confidence stated in the request; MASSIVE, LEDGAR) vs the prompt | −2.3 [−4.1, −0.6] at 5 per class, −1.2 [−2.5, +0.0] at 100: below the prompt at both budgets; Jev returned the stated label on 86.5 to 97.2% of items, so it inherited the classifier's errors | Secondary, Run 4 |
| Pool redraws, Run 2 items (6 draws): G5 product vs the prompt; 5-per-class prompt vs bare | −1.0 to +0.75 (above −1.5 in 6 of 6, above 0 in 3 of 6: sign not stable); +2.0 to +5.3 (sign stable in 6 of 6) | Secondary, Run 4 |
| CFPB time split (2025 pool at natural shares, 2,000 complaints from 2026): prompt vs bare; local product vs prompt | +0.05 [−0.6, +0.8]: no gain from retrieval; −10.9 [−12.2, −9.5], payday-loan recall 17.3% vs 71.6% (7 pool examples) | Secondary, Run 4 |
| Fitted prior × evidence model applied to the X3 arms without refitting | predicted the sign of every contrast (9 of 9 cells) and the deletion jump, but only about half of the two rival effects | Exploratory |

Caveats from the paper: the label-budget results (R3) rest on item-level clusters; under gold-class clustering no
Run 2 decision passes Holm, Run 3's label-budget estimates were inconclusive, and Run 4 did not show the local
combination non-inferior at 5 or 100 labels per class on three further datasets. G5's two datasets were chosen after
exploratory analysis, and its interval conditions on one pool draw: across six pool draws of its items the
difference stayed above −1.5 but changed sign, so read Run 2's +1.4 as conditional on one draw. Pools were
class-balanced except in the one time split (CFPB, a single dataset with a shifted class mix); the tasks are mostly
English single-label choices; one vendor, with one small check of a second version. All four pre-registrations were
written and run by one author; Runs 1, 2 and 4 were frozen but not deposited, Run 3 was deposited before its first
call. One Holm correction over all 29 tests (a sensitivity check) would no longer reject T8 and H3.

## Cost and privacy

**Cost is not the deciding factor.** At Jev's list price ($0.042 per million input tokens) every configuration the
paper ran cost $0.0008 to $0.13 per 1,000 decisions. Choosing kNN-20 over a bare call adds about $0.07 per 1,000
decisions on CFPB (about $70 per million) and $0.003 to $0.009 per 1,000 on intent and tool sets. Labelling,
engineering and monitoring are not counted and will usually cost more. Choose on data exposure, latency and
small-class recall.

US dollars per 1,000 decisions, from billed input tokens (exploratory):

| Data | Bare | kNN-20 prompt | Local combination | Cascade (share sent) |
|---|---|---|---|---|
| Tweets, Run 2 | $0.0196 | $0.0379 | $0.0196 | $0.0131 (34%) |
| CFPB, Run 2 | $0.0217 | $0.0917 | $0.0217 | $0.0215 (23.5%) |
| ToxicChat, Run 2 | $0.0135 | $0.0536 | $0.0135 | $0.0044 (8.2%) |
| MASSIVE, Run 3 | $0.0329 | $0.0370 | $0.0329 | not run |
| LEDGAR, Run 3 | $0.0240 | $0.0534 | $0.0240 | not run |
| MetaTool, Run 3 | $0.0199 | $0.0288 | $0.0199 | not run |
| CFPB, Run 3 | $0.0217 | $0.0905 | $0.0217 | not run |

The local combination's vendor cost is one bare call; the vote and the classifier run locally. The most expensive
configuration measured, kNN-20 drawn from 5 per class on CFPB, cost $0.129 per 1,000.

**What leaves your machine**

| Approach | Sent to the vendor per decision |
|---|---|
| Bare call, local combination | the item |
| kNN-20 prompt | the item plus up to 20 labelled pool texts (at most 3 per label; 10 for two options), such as other customers' complaints |
| Classifier-first cascade | nothing for items answered locally; **the item plus its nearest labelled pool texts** for every forwarded item |
| Labels-only shortlist | nothing when one label is shortlisted; otherwise the item with the shortlisted options |

The bare call still sends the item, which may itself be sensitive. `JevBackend` caches answers on disk under
`~/.cache/decision-levers/responses.sqlite`, keyed by a SHA-256 of the request: it stores the hash, the answers and
the token counts, not the request texts, with no expiry. Delete the file to drop it. What the vendor retains is set
by its terms, not by this package.

## Model versions

`JevBackend` raises when a response reports another model id than the pinned one. That is deliberate: every number
above is for `jev-1.13.0`. To move to a new version, rerun your held-out comparison (`examples/09`, `examples/11`)
with `--jev --model <new id>`, compare it with the pinned version, then re-pin. In the paper's one check on Jev 1.14,
point estimates stayed within 2.5 points, but not every effect met its pre-registered direction under class
clustering, and the local combination was not run on 1.14.

## Package layout

| Module | Contents |
|---|---|
| `request.py` | `Request`, `PackedRequest`, wire format, answer parsing |
| `backends.py` | `Backend` protocol, `JevBackend`, `MockBackend`, `ReplayBackend` (recorded answers); `decide_wire` for raw bodies |
| `data/` | `load_csv`; the toy dataset |
| `retrieval.py` | `LabelledPool` (`from_pairs`, `from_csv`), `TfidfRetriever`, `DenseRetriever` |
| `levers.py` | `add_examples`, `add_definitions`, `rename_to_ids`, `state_shares`, `correct_shares`, `pack_questions`; phrasing DON'Ts |
| `local.py` | `knn_vote`, `TunedLR`, `combine` (plain or `weights=`), `tempered_combine`, `PAPER_EXPONENTS`, `fit_exponents` |
| `cascade.py` | `ClassifierFirstCascade` (accuracy, recall or precision targets; cross-fitted report) |
| `shortlist.py` | `shortlist`, `decide_with_shortlist` |
| `evaluate.py` | `holdout_split`, `paired_bootstrap`, `tost`, `holm`, `null_controls`, `promotion_floor`, `compare`, `freeze`, `class_metrics` |
| `toy.py` | a 72-ticket hand-written dataset for the examples |

`examples/real_payloads/` holds the paper's flagship Banking77 request bodies and recorded answers (CC BY 4.0; see
its README).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The Banking77 texts in `examples/real_payloads/` are
CC BY 4.0 (Casanueva et al. 2020, PolyAI).
