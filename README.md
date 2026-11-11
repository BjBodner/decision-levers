# decision-levers

**Rules of thumb for decision models: what makes performance better (or worse).** Starter code for the paper's
do's and don'ts, and the evaluation steps that tell a real effect from noise.

> Independent project. Not affiliated with, funded by, or endorsed by TypeSafe. 'Jev' and 'TypeSafe' are used only to name the API this code calls.

Paper: arXiv link added on publication.

A typed decision model takes a text and a closed list of options and returns a probability for each option. The
paper asked which changes to a request make such a model more or less accurate. It screened about 60 techniques on
TypeSafe's Jev 1.13, re-tested the key claims in eight confirmatory runs, repeated the main tests on other decision
models, and checked them on four private datasets that no model can have seen. In short: wording wrappers (persona,
politeness, capitals) did not help. Information the model lacks did: retrieved labelled examples, and option names
that say what the option does, with a one-line description. Wrong information hurt, most when wrong labels all point
to one rival. With 5 labels per class, put retrieved examples in the prompt; with 100, the prompt is still the
default, and combining one plain call with local models is the route for when pool texts must not leave your
machine. Sizes differ between models, so confirm on yours. This package implements each approach and the checks.

## How a call works

One request holds the text to classify (the *state*), a question and the options. Each option has a name and can
carry a description. Extra information goes in one of three places: the question's wording, a one-line definition
in an option's description, or labelled examples there. Examples come from a labelled *pool*: before the call, on
your machine, you pick the pool texts nearest to the text (or random ones) and write them under their labels. The
answer is one probability per option.

```
 optional, on your machine, before the call:
 labelled pool --> pick the examples nearest to the text (or at random) ------------+
                                                                                    |
 +------------------------ the request you send --------------------------------+   |
 | state     "My card was declined at the shop."                                |   |
 | question  "Which team should handle this?"  <-- wording: persona, politeness |   |
 | options   Card Payments                                                      |   |
 |             examples     "card refused at till" ...  <---------------------------+
 |             description  "Failed card payments."  <-- one-line definition    |
 |           Refunds, Fraud  (names only; one box per option)                   |
 +------------------------------------------------------------------------------+
                                         |
                                         v
                                decision model API
                                         |
                                         v
             returned: Card Payments 0.86 | Refunds 0.09 | Fraud 0.05
```

What each addition did to accuracy against the *bare call* (the request with nothing added), from the paper's
Figure 2b: one fixed example per option +1.5, random pool examples +3.1, nearest pool examples +4.7 (Jev 1.13, Run
8); a one-line definition per option +11 (MetaTool) to +24 (invented names); persona or polite wording ≈ 0. A pool
with 30% wrong labels cost −13 against a clean pool, the mean of six other APIs. The rows come from different runs
and do not add up.

In code: `Request(state, question, options)` is the request, `add_examples` fills the examples rows from a
`LabelledPool`, `add_definitions` fills the descriptions, and `backend.decide(request)` returns the probabilities.

## Rules of thumb

The paper's Table 2. Effects are accuracy points on Jev 1.13 unless stated. Counts of other models are out of 7
(Run 5, shared items) or 6 (Run 8, fresh items, the stronger evidence; Run 10, private data). Status words:
*confirmed* = a planned test passed with classes resampled; *observed* = found after the fact, or passed only with
items resampled; *secondary* = an estimate planned in advance with no pass rule; *not shown* = a planned test that
did not pass (never "no effect").

| Rule | Evidence | Status | In this package |
|---|---|---|---|
| **Do 1.** Give it information it lacks. | Retrieved examples +5.9 to +11.2 (balanced pools), +7.7 on fresh items, +1.9 from a natural-share pool; 4 of 6 others on fresh items (Decider Flash −14.5), 3 of 6 on private data; two models failed on long texts | confirmed; partially transfers | `add_examples`; `examples/02` |
| **Do 2.** Name options by what they do, and add a one-line description, above all for unfamiliar, invented or brand-like names. | One-word names −10.7 against two-word names; a description +10.7 (MetaTool), +23.8 on an invented catalogue (+16.8 to +39.8 on 6 others); ids with a description −0.8 and 0.0 against names with one | secondary; names vs ids not shown | `add_definitions`, `rename_to_ids`, `map_back`; `examples/03` |
| **Do 3.** Audit your pool's labels. | 30% confusable wrong labels −7.0, harm on 6 of 6 others; a 10% audit won back about 1 | confirmed harm; audit secondary | guidance only (`LabelledPool.with_wrong_labels` simulates uniform noise; `examples/10`) |
| **Do 4.** At 5 per class, use the prompt. | +3.1 over the bare call, pools redrawn; the local shortcut trailed by 1.2 | confirmed | `add_examples`, `LabelledPool.cap(5)`; `examples/02` |
| **Do 5.** At 100 per class, prefer the prompt unless privacy rules it out. | The local combination trailed by 0.7 in both fresh-item tests and had lower small-class recall; it sends no pool text | not shown | `combine`, `tempered_combine`; `examples/05`, `examples/11` |
| **Do 6.** Confirm on your model. | Gains vary by model (SD 3.5); Jev 1.14 lost 3.7 more to wrong labels; 100 to 200 items rank near-ties right 56 to 64% | observed | `evaluate` (`holdout_split`, `null_controls`, `compare`, `freeze`, `holm`); `examples/09`, `examples/11` |
| **Don't 1.** Expect benign wrappers to help. | Persona, politeness, capitals, packed questions within ±2; persona and politeness on 4 of 7 others; a gain of about 1 is not ruled out | confirmed | `add_persona`, `add_emphasis`, `add_politeness` (to reproduce the null); `examples/10` |
| **Don't 2.** State unmeasured shares. | Shuffled shares −1.1, stale shares −11.5; stale lower on 6 of 6 others; true shares not shown to help | observed | `state_shares`, `correct_shares`; `examples/04` |
| **Don't 3.** Let wrong labels pile on a rival. | One rival vs spread: −13.8 on Jev, −2.8 to −10.1 on 6 of 6 others | confirmed | guidance only (`relabel_top` in `examples/10` spreads wrong labels over rivals) |
| **Don't 4.** Assume placement is neutral: test it on your model. | Penalty larger inside descriptions on Jev, and on 3 of 6 others; smaller on 2 | confirmed | guidance only (`add_examples` puts examples in the option descriptions; no separate block is implemented) |
| **Don't 5.** Judge rare classes by accuracy. | A cascade within 1.5 points on accuracy lost 7.3 points of toxic-class precision | observed | `evaluate.class_metrics`; `ClassifierFirstCascade(min_recall=, min_precision=)`; `examples/06`, `examples/11` |
| **Don't 6.** Treat wording as security. | A guard sentence, spotlighting and a claim field all fell to a score-blind search | observed | guidance only |

Don't 1 covers benign wrappers only. The paper did not vary format, option order, the option set, question
polarity or rule wording, and those can move a model a lot: rewording a ticket-priority rule moved Jev by 24 points
(cited in the paper). Test such changes as content. Details: [docs/what_not_to_do.md](docs/what_not_to_do.md).

## Which approach?

The label budget decides where your labelled data goes: into the prompt as examples, or into local models combined
with one bare call on your machine.

- **No labels.** Bare call with readable names; add a one-line description per option when names are ids, jargon,
  brands or invented (`examples/01`, `examples/03`). Do not state class shares you have not measured. Shares
  estimated from 100 labelled recent items came within about a point of the true shares (`examples/04`).
- **About 5 labels per class: use the prompt** (`add_examples`, `examples/02`). On fresh items, with the pool
  redrawn, it beat the bare call by 3.1 points [+1.8, +4.6] (confirmed). Two local shortcuts built on one bare call
  trailed it: × a nearest-centroid classifier −1.2, × a tuned classifier −0.6; neither was shown within 1.5
  points. Local classifiers used alone scored up to 18 points below the bare call.
- **About 20 to 100 labels per class: prefer the prompt unless privacy rules it out.** The prompt kept gaining
  (+3.6 at 20, +4.7 at 100 per class; secondary). The local combination (one bare call × a nearest-neighbour vote ×
  a tuned classifier, `combine(bare, vote, lr)`, `examples/05`) trailed the prompt by 0.7 points [−1.6, +0.1] in
  both fresh-item tests; neither showed it matching the prompt. Its mean worst-class recall was 29.7% against
  42.8% for the prompt (secondary). Its advantage is privacy: only the item goes to the vendor. If you go local,
  compare both on your own held-out data, per-class recall included (`examples/11`).
- **You must cut vendor calls.** A classifier-first cascade (`ClassifierFirstCascade`, `examples/06`) answers the
  items the local classifier is sure of and sends the rest to the prompt, with their nearest labelled examples. On
  ToxicChat it sent 8.2% of items and was no more than 1.5 points worse on accuracy (confirmed for these two
  classes), but toxic-class precision fell 7.3 points (observed). Set the threshold on recall or precision,
  cross-fitted (`cv=5`).
- **Your pool is older than your traffic.** A class-mix shift breaks local models. With a 2025 pool and 2026
  complaints, credit reporting fell from 74% of the pool to 3% of the test items, and the local combination fell
  10.9 points below the prompt. Re-estimating the shares from unlabelled 2026 items cut the gap to 2.0 (observed),
  but on a second split (2023 to 2024) it did not help. Score both approaches on recent held-out items.
- **Short intent sets, cost-bound.** A labels-only shortlist (`decide_with_shortlist`, `examples/07`) sends no pool
  text and lost about 2 points against the examples prompt on intent sets (observed); it hurt on topical text.

| Your situation | Try first | Example |
|---|---|---|
| No labels | Bare call; readable names; one-line descriptions for unclear names | `01`, `03` |
| About 5 labels per class | Retrieved examples in the prompt | `02` |
| About 100 labels per class | The prompt; the local combination if pool text must not leave your machine, checked on your data | `02`, `05`, `11` |
| Vendor calls must be cut | Classifier-first cascade, threshold on recall or precision, cross-fitted | `06` |
| Pool older than live traffic | Score both approaches on recent held-out items; check small-class recall | `11` (its split is random: hold out your most recent rows yourself) |
| Several decisions about the same text | Pack them into one request, main question last | `08` |
| Before shipping anything | Held-out, de-duplicated items; content-free controls; freeze; confirm | `09` |
| Your own CSV | Load, split without near-duplicate leaks, compare every approach | `11` |
| See a real request and answer | The paper's Banking77 worked example, replayed offline | `12` |

## Results across models

**Other decision models.** Run 5 sent seven other decision models the exact requests Jev 1.13 had answered; Run 8
sent the key tests to six of them and a general chat model, on items this study had never sent. Directions mostly
agree; sizes differ.

- **What hurts transfers best.** 30% confusable pool noise cost −12.6 points [−20.9, −4.3] on average and the
  one-rival penalty −6.3 [−8.4, −4.2], both on 6 of 6 models (observed pooled means). With 30% confusable noise,
  four models fell well below their own bare call (by 7.5 to 27.9 points): for them a noisy pool is worse than no
  pool. The one-rival penalty was smaller on every other model than on Jev (−13.8).
- **Retrieved examples** helped the average model by +5.6 [+2.0, +9.3] on Run 5's shared items (6 of 7). On fresh
  items the gain was confirmed on 4 of 6; Decider Flash was opposite (−14.5). Pooled: +1.1 [−5.8, +8.0], or +4.2
  [+0.9, +7.5] without Decider Flash.
- **Wrappers** stayed within ±2 on 4 of 7 models. **Ids with definitions** were within 2 points of names on 1 of 6,
  so readable names stay the default. **Placement** transferred only in part (larger inside descriptions on 3 of 6,
  smaller on 2). **Stale shares** lowered accuracy on all six in point estimate, but the loss was shown on only two.
- **The chat model** (read by option log-probabilities, on MetaTool only) showed the retrieval gain (+12.2), the pool
  noise harm and the one-rival penalty.

**Leakage check.** Every public benchmark may be in a vendor's training data. Runs 9 and 10 sent eight techniques
to eight models on four private synthetic datasets (KitchenOps-80, OpsDeck-24, ScotLease-25, BeamDesk-14) and on
public items. The techniques ranked much the same on both (rank correlations 0.76 to 1.00), and one of the six
other models' 48 effects changed sign. Cryptic ids, three wrong labels and pool noise hurt on private data on every
model; wrappers stayed within 1.3 points of zero. Descriptions helped more on private data (+7.6 against +1.0 on
the six others), most on invented names. Retrieved examples weakened (+1.1 public, +0.3 private on the six others;
held on 3 of 6).

**Caveat: examples misled two models on long texts.** On ScotLease (long lease clauses), Decider Flash fell from
98.9% to 28.1% with three correctly labelled examples, yet scored 96.9% when the same examples carried wrong
labels. On BeamDesk, d1-ultra fell from 90.6% to 53.2% with nearest examples and put 490 of 1,200 answers on one
team. Both did the same on public LEDGAR, so this is model behaviour, not leakage. **Test examples on your model,
on your longest texts, before you rely on them.** The private items were written by language models and may be
cleaner than real data.

## What didn't work

From the paper's Table 7 (Jev 1.13 unless stated; "not shown" never means "no effect"):

- Persona, politeness, capitals, packed unrelated questions: −0.3 to +0.3 against the bare call.
- Stating true class shares: +1.9 (observed) and −0.4 (not shown). Shuffled shares: −1.1.
- A natural-share pool: +1.9 [−0.4, +4.9] over the bare call, 1.4 below a balanced pool (not shown).
- Auditing the most-retrieved examples first instead of a random 10%: +0.3 [−0.4, +0.9] (not shown).
- Calibrating the bare call (contextual, batch): −2.2 and +0.2; retrieval stayed 10.0 and 7.5 ahead.
- The local combination at 100 per class (−0.7 in both tests) and the local shortcuts at 5 per class (−1.2, −0.6).
- Re-estimating shares after a time split (2023 to 2024): −0.1; worst-class recall 49.5% → 22.2%.
- Wording defences (guard sentence, spotlighting, claim field) against 100 score-blind attacks.
- Ids with definitions instead of names on fresh items: −2.3 [−3.8, −1.0] (not shown at a margin of 2).
- From the development screen: dropping pool items the bare call disputes (80.0% vs 80.2%), model-written
  descriptions, definitions on top of examples (85.8% → 82.8%), and a closing reminder against injection.

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
p_local = combine(bare, vote, lr)                             # ... combined locally (plain product)
p_temp = tempered_combine(bare, vote, lr)                     # tempered product, the paper's frozen exponents
```

The tempered product raises the three inputs to exponents, by default `PAPER_EXPONENTS` (bare, vote, classifier),
frozen before the paper's Run 4 and fitted with its dense features. In Run 4 it was −0.4 [−1.2, +0.3] against the
prompt (secondary). Refit the exponents on your own held-out items with `fit_exponents(bare, [votes, lrs], gold,
cv=5)` and keep the plain product unless the cross-fitted log-loss is lower.

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

## Cost and privacy

**Cost is not the deciding factor.** At Jev's billed rate ($0.042 per million input tokens), every design in the
paper's label-budget comparison cost $0.013 to $0.092 per 1,000 decisions in vendor fees. Using kNN-20 instead of a
bare call adds about $0.07 per 1,000 decisions on CFPB and $0.003 to $0.009 on the intent and tool sets. The other
models' list prices were $0.01 to $0.10 per million input tokens. Labelling, engineering and monitoring are not
counted and will usually cost more. Latency did not separate the designs (median 0.094 s bare, 0.104 s with
kNN-20, sequential calls in one session). Choose on data exposure and small-class recall.

US dollars per 1,000 decisions, tweets | complaints on Run 2's items (the paper's Figure 4):

| Approach | Vendor cost | Sent to the vendor per decision |
|---|---|---|
| Bare call | $0.020 \| $0.022 | the item |
| Examples prompt (kNN-20) | $0.038 \| $0.092 | the item plus up to 20 labelled pool texts (at most 3 per label; 10 for two options), such as other customers' complaints |
| Local combination | $0.020 \| $0.022, plus local compute | the item |
| Classifier-first cascade | $0.013 \| $0.022 (34% \| 23% of items sent) | nothing for items answered locally; **the item plus its nearest labelled pool texts** for every forwarded item |
| Labels-only shortlist | not in the figure | nothing when one label is shortlisted; otherwise the item with the shortlisted options |

The bare call still sends the item, which may itself be sensitive. `JevBackend` caches answers on disk under
`~/.cache/decision-levers/responses.sqlite`, keyed by a SHA-256 of the request: it stores the hash, the answers and
the token counts, not the request texts, with no expiry. Delete the file to drop it. What the vendor retains is set
by its terms, not by this package.

## Model versions

`JevBackend` raises when a response reports another model id than the pinned one. That is deliberate.
`PAPER_MODEL` is `jev-1.13.0`, the version behind most of the paper's numbers.

- **Jev 1.14** (`jev-1.14.0`) agreed with Jev 1.13's bare top option on 94 to 97% of items and passed all four Run
  5 tests, but lost 3.7 points more to wrong labels on the same items. So re-run your held-out comparison
  (`examples/09`, `examples/11`) with `--jev --model <new id>` on every version change, then re-pin.
- **Other models tested by the paper:** Perplexity Decider v1.2 and Decider Flash, OpenAI Decisions (gpt-6-luna),
  Microsoft-Decision-1, Liquid d1-ultra, Kev-4B (an open Jev clone) and Upstage Solar Decide (failed its Run 5
  pilot), plus OpenAI's gpt-6-luna chat model. They are not independent: Decider and Decider Flash are one family,
  and Microsoft-Decision-1 and Kev-4B share a base generation.

This package ships a backend for Jev only. To test another model, implement the `Backend` protocol (`decide`,
`decide_many`, `decide_packed`) for its API and run the same examples.

## Package layout

| Module | Contents |
|---|---|
| `request.py` | `Request`, `PackedRequest`, wire format, answer parsing |
| `backends.py` | `Backend` protocol, `JevBackend`, `MockBackend`, `ReplayBackend` (recorded answers); `decide_wire` for raw bodies |
| `data/` | `load_csv`; the toy dataset |
| `retrieval.py` | `LabelledPool` (`from_pairs`, `from_csv`, `cap`, `with_wrong_labels`), `TfidfRetriever`, `DenseRetriever` |
| `levers.py` | `add_examples`, `add_definitions`, `rename_to_ids`, `state_shares`, `correct_shares`, `pack_questions`; the wrapper DON'Ts |
| `local.py` | `knn_vote`, `TunedLR`, `combine` (plain or `weights=`), `tempered_combine`, `PAPER_EXPONENTS`, `fit_exponents` |
| `cascade.py` | `ClassifierFirstCascade` (accuracy, recall or precision targets; cross-fitted report) |
| `shortlist.py` | `shortlist`, `decide_with_shortlist` |
| `evaluate.py` | `holdout_split`, `paired_bootstrap`, `tost`, `holm`, `null_controls`, `promotion_floor`, `compare`, `freeze`, `class_metrics` |
| `toy.py` | a 72-ticket hand-written dataset for the examples |

`examples/real_payloads/` holds the paper's Banking77 worked example: request bodies and recorded answers (CC BY
4.0; see its README). Pseudocode for every approach and for the evaluation protocol:
[docs/algorithms.md](docs/algorithms.md). A fitted prior × evidence description of Jev, what it predicts and where
it fails: [docs/why_it_works.md](docs/why_it_works.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The Banking77 texts in `examples/real_payloads/` are
CC BY 4.0 (Casanueva et al. 2020, PolyAI).
