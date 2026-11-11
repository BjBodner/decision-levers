"""02: retrieved labelled neighbours in the option descriptions (the paper's kNN-20 prompt).

Do 1 and Do 4. At about 5 labels per class, on fresh items with the pool redrawn, it beat the bare call by 3.1
points [+1.8, +4.6] (Run 8, confirmed). At 80 to 100 per class it beat the bare call by 5.9 to 11.2 points (Run 1)
and by 7.7 on fresh items (Run 8), and the local combination of example 05 was not shown to match it: prefer it
unless privacy rules it out. It sends pool texts to the vendor. Gains differ by model (one model lost 14.5 points
on fresh items, and two were misled on long texts), so confirm on yours.
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
