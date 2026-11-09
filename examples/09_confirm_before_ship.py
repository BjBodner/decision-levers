"""09: confirm before you ship. Screen -> freeze -> confirm, with null controls.

1. split labelled data into pool and held-out items, keeping near-duplicates on one side;
2. measure what pure noise looks like: an identical resend and a content-free edit (A/A controls);
3. screen a lever against bare with a paired bootstrap, promoting only above the noise floor;
4. freeze the winning config (hash it), then confirm on held-out items you have not looked at.

On a cheap API a plausible screening rule promoted 12 of 53 content-free edits in the paper; this is why.
The toy set is far too small to confirm anything: the point here is the procedure.
"""
from _common import backend_from_args

from decision_levers import LabelledPool, add_examples, toy
from decision_levers.evaluate import PAPER_FLOOR, compare, freeze, holdout_split, null_controls, promotion_floor
from decision_levers.levers import add_persona

be = backend_from_args(__doc__)
rows = toy.load()
train_idx, held_idx, groups = holdout_split([r.text for r in rows], test_frac=0.6, seed=0)
pool = LabelledPool.from_pairs((rows[i].text, rows[i].label) for i in train_idx)
cut = 2 * len(held_idx) // 3
screen_idx, confirm_idx = held_idx[:cut], held_idx[cut:]
print(f"{len(train_idx)} pool rows, {len(screen_idx)} screening items, {len(confirm_idx)} confirmation items "
      f"({len(set(groups))} near-duplicate groups, none split across sides)")

items = [rows[i] for i in screen_idx]
gold = [r.label for r in items]
nulls = null_controls(be, [toy.bare_request(r.text) for r in items], gold)
for k, v in nulls.items():
    print(f"  null {k:<18} d_logp {v['d_logp']:+.4f}  d_acc {v['d_acc']:+.3f}  flip {v['flip']:.2f}")
floor = max(promotion_floor([v["d_logp"] for v in nulls.values()]), PAPER_FLOOR)   # two nulls are not enough
aa_flip = nulls["identical_resend"]["flip"]
print(f"promotion floor (gold log-prob): {floor:.2f}")

bare = lambda r: be.decide(toy.bare_request(r.text))                       # noqa: E731
arms = {"retrieved examples": lambda r: be.decide(add_examples(toy.bare_request(r.text), pool)),
        "persona prefix": lambda r: be.decide(add_persona(toy.bare_request(r.text)))}
winners = []
for name, fn in arms.items():
    res = compare(bare, fn, items, gold, clusters=[groups[i] for i in screen_idx], metric="logp",
                  floor=floor, aa_flip=aa_flip)
    print(f"  screen {name:<20} d_logp {res['estimate']:+.3f} [{res['lo']:+.3f}, {res['hi']:+.3f}] "
          f"flip {res['flip']:.2f} -> {res['decision']}")
    if res["decision"] == "promote":
        winners.append(name)

config = {"arms": winners, "k": 20, "per_label": "paper rule", "model": be.model, "confirm_items": confirm_idx}
print("frozen config hash:", freeze(config)[:16], "...")

items = [rows[i] for i in confirm_idx]
gold = [r.label for r in items]
if not winners:
    print("  nothing cleared the floor, so nothing is confirmed: the right outcome for a screen this small")
for name in winners:
    res = compare(bare, arms[name], items, gold, clusters=[groups[i] for i in confirm_idx], metric="acc")
    print(f"  confirm {name:<19} accuracy {100 * res['control']:.1f} -> {100 * res['treatment']:.1f} "
          f"({100 * res['estimate']:+.1f} [{100 * res['lo']:+.1f}, {100 * res['hi']:+.1f}]), p={res['p_sup']:.3f} -> "
          f"{'confirmed' if res['p_sup'] <= 0.05 else 'not confirmed (too few items to tell)'}")
