"""05: one bare call combined locally with a kNN vote and a tuned classifier (product of experts).

At about 100 labels per class this was no more than 1.5 points below the retrieval prompt on tweets and complaints
(non-inferior, Run 2, item-level clusters), with only the item sent to the vendor and no pool text. On four further
datasets (Run 3, class clusters) the estimate was inconclusive; it trailed by 0.8 to 1.5 points on Banking77,
CLINC150 and ToxicChat, and on CFPB its smallest-class recall trailed the prompt's by 10 to 16 points (exploratory).
Compare it with the prompt on your own held-out data (examples/11). The tempered product uses the paper's frozen
exponents (exploratory; refit them on your data with fit_exponents).
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
