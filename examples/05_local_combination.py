"""05: one bare call combined locally with a kNN vote and a tuned classifier (product of experts).

Do 5: at about 100 labels per class, prefer the prompt unless privacy rules it out. In the paper this product
trailed the retrieval prompt by 0.7 points [-1.6, +0.1] in both fresh-item tests (Runs 4 and 8), so we can't call
it a match; its mean worst-class recall was 29.7% against 42.8% (secondary). Go local for privacy or cost, not
accuracy: only the item goes to the vendor, no pool text. If privacy needs it, compare it with the prompt on your
own held-out data (examples/11). The tempered product uses the exponents frozen before Run 4 (-0.4 against the prompt there,
secondary); refit them on your data with fit_exponents.
"""
from _common import backend_from_args, show

from decision_levers import LabelledPool, TunedLR, add_examples, combine, knn_vote, tempered_combine, toy

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]
pool = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))
prior = toy.train_shares()                               # deployment class shares

lr = TunedLR(pool, toy.OPTIONS).fit()
print(f"tuned LR: C={lr.C}, temperature={lr.T:.3f}")
texts = [t.text for t in test]
bare = be.decide_many([toy.bare_request(s) for s in texts])
vote = [knn_vote(s, pool, toy.OPTIONS, k=10, prior=prior) for s in texts]
lrp = lr.predict(texts, prior=prior)

show("bare (one call)", bare, gold)
show("kNN vote (local)", vote, gold)
show("tuned LR (local)", lrp, gold)
show("bare x vote x LR", [combine(b, v, p) for b, v, p in zip(bare, vote, lrp)], gold)
show("tempered (paper exponents)", [tempered_combine(b, v, p) for b, v, p in zip(bare, vote, lrp)], gold)
show("kNN-20 prompt (sends pool text)", be.decide_many([add_examples(toy.bare_request(s), pool) for s in texts]), gold)
