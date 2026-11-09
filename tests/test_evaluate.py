import numpy as np
import pytest

from decision_levers import MockBackend, toy
from decision_levers.evaluate import (bootstrap_means, compare, content_free_edit, freeze, gold_logp, holdout_split,
                                      holm, near_duplicate_groups, null_controls, p_value, paired_bootstrap,
                                      promotion_floor, tost)

TEXTS = ["my parcel never arrived", "My parcel never arrived!", "my parcel has never arrived at all",
         "refund please", "the app crashes on start", "the app crashes on start up", "invoice is wrong"]


def test_near_duplicate_groups_are_transitive_and_contiguous():
    g = near_duplicate_groups(TEXTS, thresh=0.5)
    assert g[0] == g[1] == g[2] and g[4] == g[5]
    assert g[3] != g[0] and g[6] != g[4]
    assert sorted(set(g)) == list(range(len(set(g))))
    assert near_duplicate_groups(["a b", "c d"], thresh=0.9) == [0, 1]


def test_holdout_split_keeps_groups_whole():
    tr, te, g = holdout_split(TEXTS * 1, test_frac=0.4, seed=1, thresh=0.5)
    assert sorted(tr + te) == list(range(len(TEXTS)))
    assert not {g[i] for i in tr} & {g[i] for i in te}
    assert te


def test_bootstrap_ci_and_pvalues():
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 0.2, 400)
    r = paired_bootstrap(d, draws=4000)
    assert r["lo"] < d.mean() < r["hi"] and r["lo"] > 0 and r["p_sup"] < 0.01 and r["p_harm"] > 0.99
    cl = np.repeat(np.arange(40), 10)
    wide = paired_bootstrap(d + np.repeat(rng.normal(0, 0.3, 40), 10), clusters=cl, draws=4000)
    assert wide["hi"] - wide["lo"] > r["hi"] - r["lo"]                    # clustering widens the interval
    assert bootstrap_means([1.0, 1.0, 1.0], draws=10).tolist() == [1.0] * 10
    draws = np.array([0.01, 0.02, 0.03])
    assert p_value("sup", draws) == 0.25 and p_value("tost", draws, 0.05) == 0.25


def test_tost_equivalence():
    zero = np.zeros(200)
    assert tost(zero, margin=0.02, draws=2000)["equivalent"]
    big = np.r_[np.ones(100), np.zeros(100)]
    assert not tost(big, margin=0.02, draws=2000)["equivalent"]


def test_holm_matches_hand_computation():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_holm"] == pytest.approx(0.03) and out["a"]["reject"]
    assert out["c"]["p_holm"] == pytest.approx(0.06) and not out["c"]["reject"]
    assert out["b"]["p_holm"] == pytest.approx(0.06) and not out["b"]["reject"]


def test_promotion_floor_and_logp_clip():
    assert promotion_floor([0.0308, -0.0345, 0.01]) == pytest.approx(0.04)   # the paper's floor
    assert promotion_floor([0.04]) == pytest.approx(0.05)
    assert gold_logp({"a": 0.0}, "a") == pytest.approx(np.log(0.005))


def test_null_controls_on_mock():
    be = MockBackend()
    items = toy.load("test")
    reqs = [toy.bare_request(t.text) for t in items]
    out = null_controls(be, reqs, [t.label for t in items])
    assert set(out) == {"identical_resend", "content_free_edit"}
    assert all(0.0 <= v["flip"] <= 1.0 and abs(v["d_logp"]) < 0.5 for v in out.values())
    e = content_free_edit(reqs[0])
    assert e.question.startswith(reqs[0].question + " [ref ") and e.options == reqs[0].options


def test_compare_decisions():
    items = list(range(60))
    gold = ["a"] * 60
    good, bad = (lambda i: {"a": 0.9, "b": 0.1}), (lambda i: {"a": 0.1, "b": 0.9})
    half = lambda i: {"a": 0.9, "b": 0.1} if i % 2 else {"a": 0.1, "b": 0.9}   # noqa: E731
    assert compare(half, good, items, gold)["decision"] == "promote"
    assert compare(half, bad, items, gold)["decision"] == "harm"
    same = compare(good, good, items, gold, margin=0.02)
    assert same["decision"] == "no effect" and same["equivalent"] and same["flip"] == 0.0
    floored = compare(half, good, items, gold, metric="logp", floor=10.0)
    assert floored["decision"] == "no effect"


def test_freeze_is_stable():
    assert freeze({"b": 1, "a": [1, 2]}) == freeze({"a": [1, 2], "b": 1})
    assert freeze({"a": 1}) != freeze({"a": 2})
