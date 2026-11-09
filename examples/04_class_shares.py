"""04: class shares, stated in the request or (better) applied after the call.

In the paper: true shares beat no shares by +1.9 points (confirmatory) and beat shuffled shares by +6.2 to +6.8
(confirmatory; most of that is the cost of wrong shares). Correcting the bare answer after the call was at least
as good as stating the shares and added no tokens (exploratory). Both trade minority-class recall for accuracy.
"""
from _common import backend_from_args, show, tokens_per_call

from decision_levers import correct_shares, mean_probs, state_shares, toy
from decision_levers.levers import share_sentence

be = backend_from_args(__doc__)
test = toy.load("test")
gold = [t.label for t in test]
shares = toy.train_shares()
print(share_sentence(shares, toy.OPTIONS))

before = be.stats["input_tokens"]
bare = be.decide_many([toy.bare_request(t.text) for t in test])
show("bare", bare, gold, tokens_per_call(be, before, len(test)))
before = be.stats["input_tokens"]
stated = be.decide_many([state_shares(toy.bare_request(t.text), shares) for t in test])
show("true shares stated", stated, gold, tokens_per_call(be, before, len(test)))
show("bare x shares (post hoc)", [correct_shares(p, shares) for p in bare], gold)
pbar = mean_probs(bare)
show("contextual calibration (post hoc)", [correct_shares(p, shares, pbar) for p in bare], gold)
