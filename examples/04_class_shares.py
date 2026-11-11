"""04: class shares, stated in the request or (better) applied after the call.

Don't 2: state unmeasured shares. In the paper true shares were not shown to help (+1.9 observed in Run 2, -0.4
not shown in Run 8), while wrong shares hurt: shuffled shares cost 6.2 to 6.8 points against true ones, and last
year's shares 11.5 (observed). Correcting the bare answer after the call beat stating the shares by 1.3 and 2.0
points on development data and adds no tokens (observed). Both routes lost 5 to 8 points under a shifted class mix.
If you use shares, measure them first: shares from 100 labelled recent items came within about a point.
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
