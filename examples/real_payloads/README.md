# Real payloads

`flagship_banking77.json` is the paper's flagship worked example (Figure 4; Appendix M), copied unchanged from the
research project's `paper/worked_examples/` (SHA-256 `98168df1…b107303e`). It holds one Banking77 test question
(`b77full-test-2966`, *"Top up on my Google Pay isn't working."*, gold `apple_pay_or_google_pay`) sent to Jev 1.13
three ways in the paper's Run 1:

| Request | What changed | Recorded answer |
|---|---|---|
| `1_bare` | all 77 option descriptions `null` | `top_up_failed` 0.68, gold 0.30 (wrong) |
| `2_retrieved_examples` | three retrieved, correctly labelled examples under `apple_pay_or_google_pay` (TF-IDF kNN-20) | gold 0.99 (right) |
| `3_top3_relabelled` | the same three texts under `pending_top_up`, `contactless_not_working`, `passcode_forgotten` | `pending_top_up` 0.82, gold 0.01 (wrong) |

Each entry has the exact `wire_request` body, the billed input tokens and the stored probabilities (the ten largest
plus one summed row for the rest). The `sample` block records how the item was drawn: at random, with a declared
seed, from the 55 of 800 shared requests where bare was wrong, the examples right and the relabelled request wrong.
It illustrates the mechanism; the effect sizes come from all requests (net +11.2 points for examples, −33.5 for
the wrong labels), not from this one item. Internal arm and file names in the `provenance`, `arm` and
`cache_observed_by_techs` fields refer to the research code, to be released with the paper.

```bash
python examples/12_replay_flagship.py          # offline: payload diffs, recorded answers, byte-for-byte rebuild
python examples/12_replay_flagship.py --jev    # also re-sends the three bodies (needs TYPESAFE_API_KEY)
```

Re-sending costs about 3,200 input tokens, roughly $0.0001 at $0.042 per million. Jev is not bit-for-bit
deterministic (identical resends kept the argmax for 97.3 to 99.6% of requests), so compare the argmax and the rough
size of the gold probability, not the second decimal. The model id is pinned (`jev-1.13.0`): if TypeSafe retires it,
the request fails instead of silently using another version.

**Licence.** Banking77: Casanueva et al. 2020, *Efficient Intent Detection with Dual Sentence Encoders* (PolyAI),
CC BY 4.0. The texts in this file are Banking77 test and training rows, shown unmodified inside the request bodies.
The answers are outputs of TypeSafe's Jev 1.13 recorded by the author. This folder is not covered by the
package's Apache-2.0 licence for the dataset text; see `NOTICE`.
