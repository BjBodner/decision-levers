"""05: one bare call combined locally with a kNN vote and a tuned classifier (product of experts).

At about 100 labels per class on topical text this was non-inferior to the retrieval prompt in the paper
(+1.4 points, confirmatory on two datasets), at about half the tokens and with NO pool text sent to the vendor.
It trailed the prompt by 0.8 to 1.5 points on short-intent and moderation sets (exploratory).
"""
from _common import backend_from_args, show

from decision_levers import LabelledPool, TunedLR, add_examples, combine, knn_vote, toy

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
show("kNN-20 prompt (sends pool text)", be.decide_many([add_examples(toy.bare_request(s), pool) for s in texts]), gold)
