from decision_levers import (LabelledPool, Request, add_definitions, add_examples, correct_shares, map_back,
                             mean_probs, pack_questions, rename_to_ids, select_examples, state_shares)
from decision_levers.levers import (EMPHASIS, PERSONA, POLITENESS, add_emphasis, add_persona, add_politeness,
                                    default_per_label, relabel_top, share_sentence)

POOL = LabelledPool.from_pairs([
    ("charged twice on my card", "Billing"), ("invoice shows wrong price", "Billing"),
    ("charged again this month", "Billing"), ("charged for nothing", "Billing"),
    ("parcel never arrived", "Shipping"), ("tracking stopped for my parcel", "Shipping"),
])
REQ = Request("I was charged twice", "Which team?", {"Billing": None, "Shipping": "Deliveries"})


def test_pool_cap_shares_and_noise():
    assert len(POOL.cap(1)) == 2
    assert POOL.shares() == {"Billing": 4 / 6, "Shipping": 2 / 6}
    noisy = POOL.with_wrong_labels(0.5, ["Billing", "Shipping"], seed=3)
    assert sum(a != b for a, b in zip(noisy.labels, POOL.labels)) == 3
    assert noisy.texts == POOL.texts


def test_select_examples_rank_order_and_cap():
    shown = select_examples("charged twice", POOL, k=20, per_label=3)
    assert shown[0] == ("Billing", "charged twice on my card")
    assert sum(lab == "Billing" for lab, _ in shown) == 3
    assert default_per_label(2) == 10 and default_per_label(5) == 3


def test_add_examples_format_replaces_description():
    r = add_examples(REQ, POOL, k=2, per_label=3)
    assert r.options["Billing"].startswith('Examples: "charged twice on my card"; "')
    assert r.options["Shipping"] == "Deliveries"          # no example retrieved: description kept
    assert list(r.options) == list(REQ.options)          # option order unchanged
    assert REQ.options["Billing"] is None                # input not mutated


def test_definitions_and_ids():
    r = add_definitions(REQ, {"Billing": "Charges and invoices."})
    assert r.options == {"Billing": "Charges and invoices.", "Shipping": "Deliveries"}
    ids, back = rename_to_ids(r)
    assert list(ids.options) == ["option_01", "option_02"]
    assert ids.options["option_01"] == "Charges and invoices."
    assert map_back({"option_01": 0.8, "option_02": 0.2}, back) == {"Billing": 0.8, "Shipping": 0.2}


def test_share_sentence_matches_paper_format():
    s = share_sentence({"A": 0.12, "B": 0.856, "C": 0.024}, ["A", "B", "C"])
    assert s == "Approximate share of past cases by option: B 86%, A 12%, C 2%."
    r = state_shares(REQ, {"Billing": 0.7, "Shipping": 0.3})
    assert r.question == "Which team? Approximate share of past cases by option: Billing 70%, Shipping 30%."


def test_correct_shares_variants():
    p = {"a": 0.5, "b": 0.5}
    assert correct_shares(p, {"a": 0.8, "b": 0.2}) == {"a": 0.8, "b": 0.2}
    cc = correct_shares(p, {"a": 0.5, "b": 0.5}, train_shares={"a": 0.25, "b": 0.75})
    assert abs(cc["a"] - 0.75) < 1e-12 and abs(sum(cc.values()) - 1) < 1e-12
    assert correct_shares({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == {"a": 0.5, "b": 0.5}
    assert mean_probs([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0}]) == {"a": 0.5, "b": 1e-4}


def test_pack_questions_main_last():
    other = Request(REQ.state, "Is the customer upset?")
    pk = pack_questions(REQ, {"upset": other})
    assert list(pk.to_wire("m")["questions"]) == ["upset", "q"]
    assert list(pack_questions(REQ, {"upset": other}, main_last=False).questions) == ["q", "upset"]


def test_phrasing_prefixes_are_the_paper_texts():
    assert add_persona(REQ).question == PERSONA + " Which team?"
    assert add_emphasis(REQ).question.startswith("IMPORTANT: THINK CAREFULLY")
    assert add_politeness(REQ).question == POLITENESS + " Which team?"
    assert EMPHASIS.isupper()


def test_relabel_top_changes_only_the_first_n():
    shown = [("Billing", "t1"), ("Billing", "t2"), ("Shipping", "t3"), ("Shipping", "t4")]
    out = relabel_top(shown, ["Billing", "Shipping", "Refund"], n=3, seed="x")
    assert all(a[0] != b[0] for a, b in zip(shown[:3], out[:3])) and out[3] == shown[3]
    assert [t for _, t in out] == [t for _, t in shown]
