# decision-levers

**Information, not instructions.** Tested prompting levers for typed decision models, and the ones that do nothing.

> Independent project. Not affiliated with, funded by, or endorsed by TypeSafe. 'Jev' and 'TypeSafe' are used only to name the API this code calls.

Paper: link added on publication.

A typed decision model takes a text, a closed list of options and an instruction, and returns a probability for
each option. It answers from two sources: what it already knows, and what the request tells it. Edits that add no
information (a persona, politeness, capitals, unrelated questions packed alongside) did nothing measurable. Edits
that add information (retrieved labelled examples, one-line definitions, true class shares) helped, and wrong
information (mislabelled examples, shuffled shares) hurt. With enough labels, the same combination of prior and
evidence can be done outside the prompt, by multiplying one bare call by local models. This package implements
each approach the paper tested on TypeSafe Jev 1.13, plus the evaluation protocol that tells real effects from
noise.

## Which approach?

| Your situation | Try first | Example |
|---|---|---|
| No labels | Bare request; add one-line definitions if option names are ids or jargon; correct for known class shares after the call | `01`, `03`, `04` |
| About 5 labels per class | Retrieved examples in the prompt | `02` |
| About 100 labels per class, topical text | One bare call combined locally with a kNN vote and a tuned classifier, or a classifier-first cascade | `05`, `06` |
| Short, balanced intent sets, cost-bound | Labels-only shortlist (most items need no call) | `07` |
| Several decisions about the same text | Pack them into one request, main question last | `08` |
| Before shipping anything | Held-out, de-duplicated items; null controls; freeze; confirm | `09` |

**DO**

1. **Add information, not wording.** Labelled neighbours, definitions and true class shares are what moved the model.
2. **At about 5 labels per class, put retrieved examples in the prompt.** They beat bare Jev by 2 to 4 points and a tuned classifier by 16 or more.
3. **At about 100 labels per class, try combining locally first** (`combine(bare, vote, lr)`) or a classifier-first cascade. The combination matched the retrieval prompt on topical text without sending pool text; the cascade kept accuracy while sending 8% of items.
4. **Give ids and jargon one-line definitions.** Cryptic ids went from 2.3% to 84.4%.
5. **Confirm before you ship.** De-duplicate the holdout, include content-free controls, freeze, then test.

**DON'T**

1. **Don't tune wording.** Persona, politeness and capitals stayed within ±2 points.
2. **Don't trust pool labels blindly.** Three wrong nearest labels cost 13 to 34 points; one planted twin cost up to 25.
3. **Don't state shuffled or stale class shares.** Prefer correcting the bare answer after the call, with current shares.
4. **Don't pick a cascade threshold on accuracy when positives are rare.** It kept accuracy but lost 7.3 points of precision.
5. **Don't rely on prompt wording for security.** A guard sentence still let 93% of adaptive attacks through.

Details and evidence levels for each DON'T: [docs/what_not_to_do.md](docs/what_not_to_do.md). Pseudocode for
every approach and for the protocol: [docs/algorithms.md](docs/algorithms.md).

## Quickstart

```bash
pip install -e ".[dev]"            # or: pip install -e ".[dense]" for dense retrieval (fastembed)
python examples/02_retrieved_examples.py           # offline mock backend, free
pytest
```

```python
from decision_levers import LabelledPool, MockBackend, Request, add_examples, combine, knn_vote, TunedLR

pool = LabelledPool.from_pairs([("I was charged twice", "Billing"), ("my parcel never came", "Shipping"),
                                ("invoice shows the wrong plan", "Billing"), ("tracking stopped updating", "Shipping")])
req = Request(state="I was charged twice on this invoice", question="Which team should handle this ticket?",
              options=["Billing", "Shipping"])
backend = MockBackend()                 # JevBackend(PAPER_MODEL) for the real API

p_prompt = backend.decide(add_examples(req, pool))           # examples in the prompt (sends pool text)
bare = backend.decide(req)                                    # or: one bare call ...
p_local = combine(bare, knn_vote(req.state, pool, req.options),           # ... combined locally
                  TunedLR(pool, list(req.options)).fit().predict([req.state])[0])
```

To use the real API, set `TYPESAFE_API_KEY` (or `JEV_API_KEY`) and pin the model:

```python
from decision_levers import JevBackend, PAPER_MODEL
backend = JevBackend(PAPER_MODEL)       # "jev-1.13.0"; a response from another version raises
```

Every example takes `--mock` (default, offline) or `--jev` (paid; responses are cached on disk, so reruns are free).
The mock is a keyword-overlap fake built to show the mechanics; its numbers are not evidence.

## Results from the paper

All on Jev 1.13. **Confirmatory** = pre-registered, frozen by a hash before any call, Holm-corrected; Run 1 used
held-out test items, Run 2 items never sent in any request. **Exploratory** = development screens or analyses after
unsealing. Differences in accuracy points, 90% intervals in brackets.

| Lever | Result | Evidence |
|---|---|---|
| Retrieved examples (kNN-20, 80 to 100 labels per class) vs bare | +11.2 [+7.5, +15.2] Banking77; +5.9 [+4.2, +7.6] CLINC150; +8.7 [+7.5, +10.0] financial tweets | Confirmatory, Run 1 |
| Retrieved examples at 5 labels per class vs a tuned prior-corrected classifier / vs bare | tweets +17.3 / +3.6; CFPB +16.1 / +2.1 | Confirmatory, Run 2 |
| Retrieval prompt vs a fixed kNN vote (Banking77 + CLINC150) | +3.0 [+2.4, +3.7]: the prompt is better, not equivalent | Confirmatory, Run 1 |
| One bare call × vote × tuned classifier vs the kNN-20 prompt (tweets + CFPB) | +1.4 [+0.3, +2.5], non-inferior | Confirmatory, Run 2 |
| Same combination on Banking77, CLINC150, ToxicChat | −0.8 to −1.5, trails the prompt | Exploratory |
| Classifier-first cascade, ToxicChat, 8.2% of items sent | −0.2 [−0.8, +0.4], non-inferior on accuracy | Confirmatory, Run 2 |
| Same cascade, toxic-class precision | −7.3 [−13.8, −1.1] | Exploratory |
| Definitions on readable tool names (MetaTool) | +5.8 [+4.3, +7.1] | Confirmatory, Run 1 |
| Cryptic ids + definitions vs names + definitions | −0.5 [−1.5, +0.4], non-inferior (ids alone: 2.3%) | Confirmatory, Run 1 |
| True class shares vs none | +1.9 [+1.1, +2.7] | Confirmatory, Run 2 |
| True vs shuffled class shares | +6.8 tweets, +6.2 CFPB | Confirmatory, Run 1 |
| Shares applied after the call vs stated | +1.3 tweets, +2.0 CFPB | Exploratory |
| Wrong labels on the 3 nearest shown examples | −33.5 Banking77, −13.4 tweets | Confirmatory, Run 1 |
| Rank effect at matched evidence (nearest vs random ranks relabelled) | +1.2 [−0.4, +2.7], not confirmed | Confirmatory, Run 2 |
| Persona, capitals, politeness | each within ±2; intervals inside [−1.3, +0.5] | Confirmatory, Run 1 |
| Seven unrelated questions packed with the main one | within ±2 | Confirmatory, Run 1 |
| Three related questions packed vs separate (CFPB) | Δ 0.0 on the main question; 51% fewer input tokens | Exploratory |
| Labels-only shortlist vs kNN-20 prompt (Banking77) | 91.4 vs 93.7; 62 vs 1,117 tokens; 80% of items without a call | Exploratory |
| Naive screening rule on content-free edits | promoted 12 of 53 | Exploratory |

Caveats from the paper: the equivalence decisions hold for accuracy averaged over two instruction wordings with
item-level clusters; no Run 2 decision survives clustering by gold class; each Run 2 test rests on one or two
datasets; one model version.

## Cost and privacy

- **The retrieval prompt sends pool texts to the vendor.** Every request carries up to 20 labelled examples from
  your data. On tweets it roughly doubled input tokens (about 470 to 990), i.e. about $0.020 to $0.042 per 1,000
  decisions at the paper's price (exploratory).
- **The local combination does not.** `combine(bare, vote, lr)` sends only the text being decided; the pool stays
  on your machine. It used about half the tokens of the retrieval prompt.
- The cascade and the shortlist send nothing for the items they answer locally.
- `JevBackend` caches responses on disk under `~/.cache/decision-levers/` (keyed by the request hash). Delete
  the folder to drop stored texts and answers.

## Package layout

| Module | Contents |
|---|---|
| `request.py` | `Request`, `PackedRequest`, wire format, answer parsing |
| `backends.py` | `Backend` protocol, `JevBackend`, `MockBackend` |
| `retrieval.py` | `LabelledPool`, `TfidfRetriever`, `DenseRetriever` |
| `levers.py` | `add_examples`, `add_definitions`, `rename_to_ids`, `state_shares`, `correct_shares`, `pack_questions`; phrasing DON'Ts |
| `local.py` | `knn_vote`, `TunedLR`, `combine` |
| `cascade.py` | `ClassifierFirstCascade` |
| `shortlist.py` | `shortlist`, `decide_with_shortlist` |
| `evaluate.py` | `holdout_split`, `paired_bootstrap`, `tost`, `holm`, `null_controls`, `promotion_floor`, `compare`, `freeze` |
| `toy.py` | a 72-ticket hand-written dataset for the examples |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
