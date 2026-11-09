"""v0.2: cross-fitted cascade thresholds, the precision target, tempered combine, load_csv, replay."""
import json
import math
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import decision_levers
from decision_levers import (PAPER_EXPONENTS, BackendError, ClassifierFirstCascade, LabelledPool, ReplayBackend,
                             Request, combine, fit_exponents, load_csv, tempered_combine)
from decision_levers.cascade import _folds
from decision_levers.evaluate import class_metrics

EX = Path(__file__).resolve().parents[1] / "examples"
FLAGSHIP = EX / "real_payloads" / "flagship_banking77.json"


# ------------------------------------------------------------------------------------------------ cascade

def _noisy_calibration(n=200, seed=3):
    """Local answers right with probability = their confidence; the model is always right."""
    rng = random.Random(seed)
    local, gold = [], []
    for _ in range(n):
        c = rng.uniform(0.5, 1.0)
        right = rng.random() < c
        local.append({"a": c, "b": 1 - c} if right else {"b": c, "a": 1 - c})
        gold.append("a")
    return local, [{"a": 1.0, "b": 0.0}] * n, gold


def test_folds_cover_every_item_once_and_keep_groups_whole():
    groups = [i // 3 for i in range(30)]                 # 10 groups of 3
    folds = _folds(30, 4, groups, random.Random(0))
    flat = np.concatenate(folds)
    assert sorted(flat.tolist()) == list(range(30))
    for f in folds:
        assert {groups[i] for i in f}.isdisjoint({groups[i] for g in folds if g is not f for i in g})
    assert max(map(len, folds)) - min(map(len, folds)) <= 3


def test_crossfit_report_is_less_optimistic_than_in_sample():
    local, model, gold = _noisy_calibration()
    cas = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.0, cv=5, repeats=3, seed=1)
    cf = cas.report["crossfit"]
    assert cas.report["optimistic"] is True
    assert cas.report["accuracy"] == 1.0                 # in sample, the target is met exactly
    assert cf["accuracy"] < cas.report["accuracy"]       # out of fold it is not: the in-sample number flatters
    assert not cf["meets_target"]
    assert cf["folds"] == 5 and cf["repeats"] == 3 and len(cf["fold_thresholds"]) == 15
    assert 0.0 <= cf["share_sent"] <= 1.0
    again = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.0, cv=5, repeats=3, seed=1)
    assert again.report["crossfit"] == cf                # seeded: reproducible
    assert cas.threshold == ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.0).threshold


def test_crossfit_with_groups_and_errors():
    local, model, gold = _noisy_calibration(60)
    cas = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.02, cv=3, groups=[i // 4 for i in range(60)])
    assert len(cas.report["crossfit"]["fold_thresholds"]) == 3
    with pytest.raises(ValueError):
        ClassifierFirstCascade().fit(local, model, gold, cv=1)
    with pytest.raises(ValueError):
        ClassifierFirstCascade().fit(local, model, gold, cv=3, groups=[0] * 60)     # one group: one fold
    with pytest.raises(ValueError):
        ClassifierFirstCascade().fit(local, model, gold[:-1])


def _false_positive_case():
    # 5 positives the local model gets right (conf .95); 10 negatives it calls positive (conf .70);
    # 35 negatives it gets right (conf .99). The model is always right.
    gold = ["pos"] * 5 + ["neg"] * 45
    local = ([{"pos": 0.95, "neg": 0.05}] * 5 + [{"pos": 0.70, "neg": 0.30}] * 10
             + [{"neg": 0.99, "pos": 0.01}] * 35)
    model = [{"pos": 1.0, "neg": 0.0}] * 5 + [{"neg": 1.0, "pos": 0.0}] * 45
    return local, model, gold


def test_precision_target_sends_the_false_positives():
    local, model, gold = _false_positive_case()
    acc = ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=0.25, positive="pos")
    assert acc.report["share_sent"] == 0.0 and acc.report["precision"] == pytest.approx(5 / 15)
    prec = ClassifierFirstCascade().fit(local, model, gold, min_precision=0.9, positive="pos")
    assert prec.report["feasible"] and prec.report["precision"] >= 0.9
    assert prec.report["share_sent"] == pytest.approx(10 / 50)
    assert prec.report["recall"] == 1.0
    assert prec.report["full_model"]["precision"] == 1.0
    sent = prec.route(local)
    assert sent[5:15].all() and not sent[:5].any() and not sent[15:].any()
    both = ClassifierFirstCascade().fit(local, model, gold, min_precision=0.9, min_recall=1.0, positive="pos")
    assert both.report["share_sent"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ClassifierFirstCascade().fit(local, model, gold, min_precision=0.9)


def test_precision_target_with_crossfit_and_unreachable_floor():
    local, model, gold = _false_positive_case()
    cas = ClassifierFirstCascade().fit(local, model, gold, min_precision=0.9, positive="pos", cv=5)
    assert {"precision", "recall", "meets_target"} <= set(cas.report["crossfit"])
    # a model that never says "pos" has undefined precision, counted as 0: the floor cannot be met
    never = [{"neg": 1.0, "pos": 0.0}] * 50
    none = ClassifierFirstCascade().fit([{"neg": 0.99, "pos": 0.01}] * 50, never, gold, min_precision=0.5,
                                        positive="pos")
    assert not none.report["feasible"] and math.isinf(none.threshold) and none.report["share_sent"] == 1.0


# ------------------------------------------------------------------------------------------------ combine

def test_tempered_combine_weights():
    bare, vote, lr = {"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}, {"a": 0.7, "b": 0.3}
    assert combine(bare, vote, lr, weights=(1, 1, 1)) == pytest.approx(combine(bare, vote, lr))
    assert combine(bare, vote, lr, weights=(1, 0, 0), eps=0.0) == pytest.approx(bare)
    a, b, c = PAPER_EXPONENTS
    raw = {o: (bare[o] + 1e-3) ** a * (vote[o] + 1e-3) ** b * (lr[o] + 1e-3) ** c for o in bare}
    z = sum(raw.values())
    expect = {o: v / z for o, v in raw.items()}
    assert tempered_combine(bare, vote, lr) == pytest.approx(expect)
    assert combine(bare, vote, lr, weights="paper") == pytest.approx(expect)
    assert PAPER_EXPONENTS == (0.3579, 0.1310, 0.6201)
    assert [round(x, 3) for x in PAPER_EXPONENTS] == [0.358, 0.131, 0.620]
    assert list(tempered_combine({"b": 0.9, "a": 0.1}, vote, lr)) == ["b", "a"]
    zeros = combine(bare, vote, lr, weights=(0, 0, 0))
    assert zeros == pytest.approx({"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        combine(bare, vote, weights=(1, 1, 1))
    with pytest.raises(ValueError):
        combine(bare, vote, lr, weights=(1, -0.1, 1))
    with pytest.raises(ValueError):
        combine(bare, vote, weights="paper")
    with pytest.raises(ValueError):
        combine(bare, vote, lr, weights="fitted")


def _stack_data(n=300, seed=0):
    """Bare is uninformative noise, the vote is informative, the LR is half-informative."""
    rng = np.random.default_rng(seed)
    opts = ["x", "y", "z"]
    bare, vote, lr, gold = [], [], [], []
    for _ in range(n):
        g = int(rng.integers(3))
        def dist(sharp):
            v = rng.dirichlet(np.ones(3))
            v = (1 - sharp) * v
            v[g] += sharp
            return dict(zip(opts, v.tolist()))
        bare.append(dist(0.0))
        vote.append(dist(0.7))
        lr.append(dist(0.3))
        gold.append(opts[g])
    return bare, vote, lr, gold


def test_fit_exponents_by_cross_validation():
    bare, vote, lr, gold = _stack_data()
    fit = fit_exponents(bare, [vote, lr], gold, cv=5, groups=[i // 2 for i in range(len(gold))])
    a, b, c = fit["weights"]
    assert min(a, b, c) >= 0 and b > a and b > c
    assert len(fit["fold_weights"]) == 5 and fit["n"] == 300 and fit["options"] == ["x", "y", "z"]
    assert fit["crossfit"]["log_loss"] < fit["plain"]["log_loss"]       # tempering pays out of fold here
    assert 0 <= fit["crossfit"]["accuracy"] <= 1
    assert fit_exponents(bare, [vote, lr], gold, cv=5)["weights"] == pytest.approx(fit["weights"])
    p = combine(bare[0], vote[0], lr[0], weights=fit["weights"])
    assert sum(p.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_exponents(bare, [vote, lr[:-1]], gold)
    with pytest.raises(ValueError):
        fit_exponents(bare, [vote, lr], ["w"] * len(gold))
    with pytest.raises(ValueError):
        fit_exponents([{"x": 1.0, "y": 0.0}, {"y": 0.5, "x": 0.5}], [], ["x", "y"], cv=2)


# ------------------------------------------------------------------------------------------------ load_csv

def test_load_csv_and_pool_from_csv(tmp_path):
    p = tmp_path / "mine.csv"
    p.write_text("body,queue,split\n"
                 "  charged twice ,Billing,train\n"
                 "parcel lost,Shipping,train\n"
                 ",Billing,train\n"                                   # empty text: skipped
                 "\"refund, please\",Refund,test\n"
                 "invoice wrong,Billing,train\n", encoding="utf-8-sig")  # with a byte-order mark
    rows = load_csv(p, "body", "queue")
    assert rows == [("charged twice", "Billing"), ("parcel lost", "Shipping"), ("refund, please", "Refund"),
                    ("invoice wrong", "Billing")]
    assert load_csv(p, "body", "queue", where={"split": "test"}) == [("refund, please", "Refund")]
    pool = LabelledPool.from_csv(p, "body", "queue", where={"split": "train"}, per_class=1)
    assert pool.labels == ["Billing", "Shipping"] and pool.texts == ["charged twice", "parcel lost"]
    with pytest.raises(ValueError):
        load_csv(p, "body", "queue", skip_empty=False)
    with pytest.raises(KeyError, match="header has"):
        load_csv(p, "text", "queue")
    t = tmp_path / "mine.tsv"
    t.write_text("text\tlabel\na, b\tX\n", encoding="utf-8")
    assert load_csv(t) == [("a, b", "X")]
    assert decision_levers.data.load_csv is load_csv


def test_bundled_example_csv_loads():
    rows = load_csv(EX / "data" / "hr_requests.csv", "message", "team")
    assert len(rows) == 60 and {lab for _, lab in rows} == {"Payroll", "Time off", "IT support", "Benefits"}


def test_class_metrics():
    ans = [{"a": 0.9, "b": 0.1}, {"a": 0.6, "b": 0.4}, {"b": 0.8, "a": 0.2}, {"a": 0.7, "b": 0.3}]
    m = class_metrics(ans, ["a", "b", "b", "a"])
    assert m["accuracy"] == 0.75 and m["recall"] == {"a": 1.0, "b": 0.5}
    assert m["worst_class"] == "b" and m["worst_class_recall"] == 0.5
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert math.isnan(class_metrics(ans, ["a", "b", "b", "a"], min_items=3)["worst_class_recall"])


# ------------------------------------------------------------------------------------------------ replay

def test_replay_backend_and_decide_wire():
    doc = json.loads(FLAGSHIP.read_text(encoding="utf-8"))
    assert list(doc["requests"]) == ["1_bare", "2_retrieved_examples", "3_top3_relabelled"]
    sides = list(doc["requests"].values())
    rb = ReplayBackend([(s["wire_request"], {"q": {"top_up_failed": 0.5}}) for s in sides[:1]])
    assert rb.model == "jev-1.13.0"
    assert rb.decide_wire(sides[0]["wire_request"])["q"] == {"top_up_failed": 0.5}
    w = sides[0]["wire_request"]
    req = Request(w["state"], w["questions"]["q"]["instructions"], w["questions"]["q"]["criteria"])
    assert rb.decide(req) == {"top_up_failed": 0.5}                    # the package rebuilds the same body
    with pytest.raises(BackendError, match="byte for byte"):
        rb.decide_wire(sides[1]["wire_request"])
    with pytest.raises(BackendError, match="pinned"):
        rb.decide_wire({**w, "model": "jev-1.14.0"})
    swapped = dict(w)
    swapped["questions"] = {"q": dict(reversed(list(w["questions"]["q"].items())))}
    with pytest.raises(BackendError):                                   # same content, different key order
        rb.decide_wire(swapped)


def test_replay_example_offline(monkeypatch, capsys):
    script = EX / "12_replay_flagship.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--mock"])
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "rebuilds all three bodies byte for byte: True" in out
    assert "argmax agreement with the recorded answers 3 of 3" in out
    for s in ("-> top_up_failed (wrong", "-> apple_pay_or_google_pay (right", "-> pending_top_up (wrong"):
        assert s in out
    assert "~ pending_top_up: null" in out


def test_replay_example_jev_without_key_sends_nothing(monkeypatch, capsys):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("must not build a JevBackend without a key")

    for k in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(decision_levers, "JevBackend", Boom)
    script = EX / "12_replay_flagship.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--jev"])
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "nothing was sent" in out and "3 of 3" in out


def test_replay_example_jev_resends_through_pinned_backend(monkeypatch, capsys):
    doc = json.loads(FLAGSHIP.read_text(encoding="utf-8"))
    recorded = {json.dumps(s["wire_request"]): {o: p for o, p in s["probabilities"].items()
                                                 if not o.startswith("(other ")}
                for s in doc["requests"].values()}
    sent = []

    class FakeJev:
        def __init__(self, model):
            assert model == "jev-1.13.0"

        def decide_wire(self, wire):
            sent.append(wire)
            return {"q": recorded[json.dumps(wire)]}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-key")
    monkeypatch.setattr(decision_levers, "JevBackend", FakeJev)
    script = EX / "12_replay_flagship.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--jev"])
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert len(sent) == 3 and "re-sent to the API now" in out
    assert "argmax agreement with the recorded answers: 3 of 3" in out
