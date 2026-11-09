"""02: retrieved labelled neighbours in the option descriptions (the paper's kNN-20 prompt).

DO this when you have few labels (about 5 per class): in the paper it beat bare Jev by 2 to 4 points and every
local classifier by about 16. At 100 per class it beat bare by 5.9 to 11.2 points (confirmatory), but see
example 05 for a cheaper way at that budget. Note that this sends pool texts to the vendor.
"""
from _common import backend_from_args, show, tokens_per_call

from decision_levers import LabelledPool, add_examples, toy

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]
pool = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))

req = add_examples(toy.bare_request(test[0].text), pool, k=20)
print("options with examples for the first item:")
for o, d in req.options.items():
    print(f"  {o:<15} {d if d else '-'}")

for name, make in [("bare", toy.bare_request),
                   ("kNN-20, 5 per class", lambda s: add_examples(toy.bare_request(s), pool.cap(5))),
                   ("kNN-20, whole pool", lambda s: add_examples(toy.bare_request(s), pool))]:
    before = be.stats["input_tokens"]
    show(name, be.decide_many([make(t.text) for t in test]), gold, tokens_per_call(be, before, len(test)))
