"""03: names and one-line descriptions. Compares four ways to present the options: readable names, names with a
one-line description, cryptic ids, and ids with a description.

Do 2: name options by what they do, and add a one-line description, above all for unfamiliar, invented or
brand-like names. In the paper (Run 9, secondary) a description added 10.7 points to MetaTool's tool names and 23.8
on an invented catalogue; names cut to an invented brand word fell to 5.0%, close to the 2.1% of cryptic ids. Ids
with a description came within a point of names with one there, but were 2.3 points below on fresh items (Run 8,
not shown within 2), so ids with descriptions are one option, not the default.
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
