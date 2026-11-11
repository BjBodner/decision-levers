"""08: several questions about one state in ONE request.

In the paper three related questions about one complaint, packed, left the main answer unchanged (80.75% both
ways, observed) and saved 51% of input tokens; seven unrelated companion questions left the main answer within +-2
points on fresh items (Run 8). Put the main question last.
"""
from _common import backend_from_args, tokens_per_call

from decision_levers import Request, pack_questions, toy

be = backend_from_args(__doc__)
test = toy.load("test")


def questions(text):
    main = toy.bare_request(text)
    others = {"upset": Request(text, "Is the customer upset or angry?"),
              "order": Request(text, "Does the message refer to a specific order or parcel?")}
    return main, others


packed = pack_questions(*questions(test[0].text))
print("packed wire body:", packed.to_wire(be.model))

before = be.stats["input_tokens"]
sep = []
for t in test:
    main, others = questions(t.text)
    sep.append({"q": be.decide(main), **{n: be.decide(r) for n, r in others.items()}})
tok_sep = tokens_per_call(be, before, len(test))
before = be.stats["input_tokens"]
pk = [be.decide_packed(pack_questions(*questions(t.text))) for t in test]
tok_pk = tokens_per_call(be, before, len(test))

for name in ("q", "upset", "order"):
    agree = sum(max(a[name], key=a[name].get) == max(b[name], key=b[name].get) for a, b in zip(sep, pk)) / len(test)
    print(f"  {name:<6} argmax agreement packed vs separate: {agree:.0%}")
print(f"  input tokens per ticket: separate {tok_sep:,.0f}, packed {tok_pk:,.0f} ({1 - tok_pk / tok_sep:.0%} saved)")
