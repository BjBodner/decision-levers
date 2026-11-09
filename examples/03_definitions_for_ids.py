"""03: one-line definitions, which rescue cryptic option ids.

DO this when option names are ids or jargon. In the paper (confirmatory) cryptic tool ids alone scored 2.3%; with
definitions 84.4%, no more than 2 points below readable names with definitions (non-inferior). Definitions written
blind on three further datasets were no more than 2 points below names on average, but LEDGAR was 2.4 below (Run 3).
Definitions on readable names added +5.8.
"""
from _common import backend_from_args, show

from decision_levers import add_definitions, map_back, rename_to_ids, toy

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]


def ids(text, defs):
    req = toy.bare_request(text)
    if defs:
        req = add_definitions(req, toy.DEFINITIONS)
    return rename_to_ids(req)          # (request with option_01.., id -> real name)


print("cryptic request:", ids(test[0].text, True)[0].options)
for name, defs in [("ids, no definitions", False), ("ids + definitions", True)]:
    pairs = [ids(t.text, defs) for t in test]
    ans = [map_back(p, back) for p, (_, back) in zip(be.decide_many([r for r, _ in pairs]), pairs)]
    show(name, ans, gold)
show("names, no definitions", be.decide_many([toy.bare_request(t.text) for t in test]), gold)
show("names + definitions", be.decide_many([add_definitions(toy.bare_request(t.text), toy.DEFINITIONS)
                                             for t in test]), gold)
