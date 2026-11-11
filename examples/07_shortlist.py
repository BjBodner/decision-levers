"""07: labels-only shortlist. Send only the labels of the 5 nearest neighbours as options, no example text; if
they all agree, answer without a call.

Observed in the paper: on Banking77 and CLINC150 it lost 2.2 and 2.4 points against the kNN-20 prompt at about 5%
of its tokens, and it sends no pool text. On topical text (tweets) it fell BELOW the bare model.
"""
from _common import backend_from_args, show, tokens_per_call

from decision_levers import LabelledPool, add_examples, decide_with_shortlist, shortlist, toy

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]
pool = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))

print("shortlist for the first item:", shortlist(test[0].text, pool, k=5))
before, calls = be.stats["input_tokens"], be.stats["calls"]
res = [decide_with_shortlist(toy.bare_request(t.text), pool, be, k=5) for t in test]
n_calls = be.stats["calls"] - calls
show(f"shortlist-5 ({1 - n_calls / len(test):.0%} without a call)", [p for p, _ in res], gold,
     tokens_per_call(be, before, len(test)))
before = be.stats["input_tokens"]
show("kNN-20 prompt", be.decide_many([add_examples(toy.bare_request(t.text), pool) for t in test]), gold,
     tokens_per_call(be, before, len(test)))
print(f"gold label inside the shortlist: {sum(t.label in shortlist(t.text, pool) for t in test) / len(test):.0%}")
