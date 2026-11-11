"""10: what NOT to do. Benign wrappers add no information, so they stayed within +-2 points; wrong labels remove
information, so they hurt.

Don't 1 and Do 3 in the paper: five personas and five politeness prefixes stayed within +-2 accuracy points of the
bare request on fresh items of four datasets (Run 3, confirmed), and so did a capitals line on fresh items (Run 8).
Wrong labels hurt: on the three nearest shown examples they cost 33.5 and 13.4 points (Run 1); relabelling one, two
and three of three correct shown examples cost 2.9, 11.4 and 38.2 (Run 3; slope confirmed); and 30% confusable wrong
labels in a pool cost 7.0 on Jev and harmed all six other models tested (Run 8). Here the wrong labels are spread
over rivals; piling them on one rival costs more (Don't 3). This covers benign wrappers only: reworded rules and
decision-relevant context do move the model.
"""
from _common import backend_from_args, show

from decision_levers import LabelledPool, select_examples, toy, with_examples
from decision_levers.evaluate import compare
from decision_levers.levers import add_emphasis, add_persona, add_politeness, relabel_top

be = backend_from_args(__doc__)
test = toy.load("test") + toy.load("calib")
gold = [t.label for t in test]
pool = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))

bare = lambda t: be.decide(toy.bare_request(t.text))                       # noqa: E731
print("phrasing (equivalence margin +-2 points; on 32 toy items the flag is an illustration, not a test):")
for name, edit in [("persona", add_persona), ("emphasis", add_emphasis), ("politeness", add_politeness)]:
    r = compare(bare, lambda t, e=edit: be.decide(e(toy.bare_request(t.text))), test, gold, margin=0.02)
    print(f"  {name:<11} accuracy change {100 * r['estimate']:+5.1f} points [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"
          f", flips {r['flip']:.0%}, equivalent: {r['equivalent']}")

print("wrong labels on the 3 nearest shown examples:")
clean, wrong = [], []
for t in test:
    shown = select_examples(t.text, pool, k=20, n_options=len(toy.OPTIONS))
    clean.append(with_examples(toy.bare_request(t.text), shown))
    wrong.append(with_examples(toy.bare_request(t.text), relabel_top(shown, toy.OPTIONS, n=3, seed=t.text)))
show("clean examples", be.decide_many(clean), gold)
show("top-3 labels wrong", be.decide_many(wrong), gold)
