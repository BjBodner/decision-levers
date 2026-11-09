import numpy as np
import pytest

from decision_levers import (ClassifierFirstCascade, LabelledPool, MockBackend, Request, TunedLR, combine,
                             decide_with_shortlist, knn_vote, shortlist, toy)
from decision_levers.shortlist import restrict

POOL = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))


def test_combine_is_a_renormalised_product():
    out = combine({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}, eps=0.0)
    assert out == pytest.approx({"a": 0.9, "b": 0.1})
    out = combine({"a": 0.2, "b": 0.8}, {"a": 0.8, "b": 0.2}, {"a": 0.5, "b": 0.5}, eps=1e-3)
    assert sum(out.values()) == pytest.approx(1.0) and out["a"] == pytest.approx(out["b"])
    assert list(combine({"b": 1.0, "a": 0.0}, {"a": 1.0})) == ["b", "a"]   # bare's options and order


def test_knn_vote_and_prior_correction():
    v = knn_vote("my parcel tracking has not updated", POOL, toy.OPTIONS, k=10)
    assert sum(v.values()) == pytest.approx(1.0) and max(v, key=v.get) == "Shipping"
    flat = {o: 0.2 for o in toy.OPTIONS}
    vp = knn_vote("my parcel tracking has not updated", POOL, toy.OPTIONS, k=10, prior=flat)
    assert sum(vp.values()) == pytest.approx(1.0)
    assert vp["Bug report"] / vp["Billing"] > v["Bug report"] / v["Billing"]   # rare class up-weighted


def test_tuned_lr_fits_and_predicts():
    lr = TunedLR(POOL, toy.OPTIONS).fit()
    assert lr.C in (0.1, 1.0, 10.0, 100.0, 1000.0) and lr.T > 0
    preds = lr.predict([t.text for t in toy.load("test")], prior=toy.train_shares())
    assert all(sum(p.values()) == pytest.approx(1.0) for p in preds)
    acc = np.mean([max(p, key=p.get) == t.label for p, t in zip(preds, toy.load("test"))])
    assert acc > 0.5


def _conf(c, lab, other="b"):
    return {lab: c, other: 1 - c}


def test_cascade_threshold_meets_accuracy_target():
    # local is right when confident (> 0.8), wrong otherwise; the model is always right
    conf = np.linspace(0.55, 0.99, 40)
    gold = ["a"] * 40
    local = [_conf(c, "a" if c > 0.8 else "b", "a" if c <= 0.8 else "b") for c in conf]
    model = [{"a": 1.0, "b": 0.0}] * 40
    cas = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.0)
    assert cas.report["accuracy"] == 1.0
    assert cas.report["share_sent"] == pytest.approx(np.mean(conf <= 0.8))
    loose = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.5)
    assert loose.report["share_sent"] < cas.report["share_sent"]
    route = cas.route(local)
    assert route.sum() == round(cas.report["share_sent"] * 40)


def test_cascade_recall_target_and_predict():
    gold = ["pos"] * 5 + ["neg"] * 35
    local = [{"neg": 0.95, "pos": 0.05}] * 40          # confident and always "neg": misses every positive
    model = [{"pos": 1.0, "neg": 0.0}] * 5 + [{"neg": 1.0, "pos": 0.0}] * 35
    acc = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.2)
    assert acc.report["share_sent"] == 0.0             # an accuracy target lets every positive be missed
    rec = ClassifierFirstCascade().fit(local, model, gold, min_recall=0.8, positive="pos")
    assert rec.report["share_sent"] == 1.0
    out, sent = rec.predict(["x"] * 3, lambda ts: local[:len(ts)], lambda t: {"pos": 1.0, "neg": 0.0})
    assert sent == 1.0 and out[0] == {"pos": 1.0, "neg": 0.0}
    with pytest.raises(ValueError):
        ClassifierFirstCascade().fit(local, model, gold, min_recall=0.5)


def test_shortlist_restrict_and_skip():
    sl = shortlist("my parcel tracking has not updated", POOL, k=5)
    assert sl[0] == "Shipping" and len(sl) == len(set(sl))
    req = toy.bare_request("x")
    assert list(restrict(req, ["Refund", "Billing"]).options) == ["Billing", "Refund"]

    class Never:
        def decide(self, r):
            raise AssertionError("must not call")

    one = LabelledPool(["Refund"] * 3, ["money back please", "refund me", "return money"])
    p, called = decide_with_shortlist(toy.bare_request("refund my money"), one, Never(), k=3)
    assert not called and p["Refund"] == 1.0 and sum(p.values()) == 1.0
    be = MockBackend()
    p, called = decide_with_shortlist(toy.bare_request("my parcel tracking"), POOL, be, k=5)
    assert called and set(p) == set(toy.OPTIONS)
    assert all(p[o] == 0.0 for o in toy.OPTIONS if o not in shortlist("my parcel tracking", POOL, 5))


def test_noul_request_through_mock():
    p = MockBackend().decide(Request("my parcel never arrived", "Did the parcel arrive?"))
    assert set(p) == {"true", "false"}
