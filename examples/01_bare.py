"""01: the bare request (option names only). Every other approach is measured against this.

    python examples/01_bare.py            # offline mock
    python examples/01_bare.py --jev      # real API
"""
from _common import backend_from_args, show, tokens_per_call

from decision_levers import toy

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]

req = toy.bare_request(test[0].text)
print("wire body of one request:", req.to_wire(be.model))

before = be.stats["input_tokens"]
answers = be.decide_many([toy.bare_request(t.text) for t in test])
print("first answer:", {k: v for k, v in answers[0].items() if v > 0})
show("bare (names only)", answers, gold, tokens_per_call(be, before, len(test)))
