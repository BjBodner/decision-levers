import json

import httpx
import pytest

from decision_levers import JevBackend, MockBackend, PackedRequest, Request
from decision_levers.backends import BackendError, request_key
from decision_levers.request import parse_answer


def test_wire_format_choice_and_noul():
    r = Request("my parcel is late", "Which team?", {"Shipping": None, "Billing": "Charges"})
    assert r.qtype == "choice"
    assert r.to_wire("jev-1.13.0") == {
        "model": "jev-1.13.0", "state": "my parcel is late",
        "questions": {"q": {"type": "choice", "instructions": "Which team?",
                            "criteria": {"Shipping": None, "Billing": "Charges"}}}}
    yn = Request("my parcel is late", "Is the customer upset?")
    assert yn.qtype == "noul" and yn.option_names() == ["true", "false"]
    assert "criteria" not in yn.to_wire("m")["questions"]["q"]


def test_list_options_and_validation():
    assert Request("s", "q", ["a", "b"]).options == {"a": None, "b": None}
    with pytest.raises(ValueError):
        Request("s", "q", qtype="choice")
    with pytest.raises(ValueError):
        PackedRequest("s", {"a": Request("other", "q")})


def test_parse_answer():
    assert parse_answer({"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}) == {"a": 0.7, "b": 0.3}
    assert parse_answer({"type": "noul", "noul": 0.25}) == {"true": 0.25, "false": 0.75}


def test_request_key_depends_on_order_and_rep():
    a = Request("s", "q", ["a", "b"]).to_wire("m")
    b = Request("s", "q", ["b", "a"]).to_wire("m")
    assert request_key("m", a) != request_key("m", b)
    assert request_key("m", a, 0) != request_key("m", a, 1)


def test_mock_is_deterministic_and_counts_tokens():
    be = MockBackend()
    r = Request("I was charged twice on my invoice", "Which team?", {"Billing": "invoice charged", "Shipping": None})
    p1, p2 = be.decide(r), be.decide(r)
    assert p1 == p2 and set(p1) == {"Billing", "Shipping"}
    assert max(p1, key=p1.get) == "Billing"
    knows = MockBackend(knowledge={"Billing": ["invoice"]}).decide(r.with_options(["Billing", "Shipping"]))
    assert max(knows, key=knows.get) == "Billing"
    assert be.stats["calls"] == 2 and be.stats["input_tokens"] > 0


def test_mock_packed_matches_separate():
    be = MockBackend()
    main = Request("parcel never arrived", "Which team?", ["Shipping", "Billing"])
    other = Request("parcel never arrived", "Is the parcel late?")
    packed = be.decide_packed(PackedRequest(main.state, {"o": other, "q": main}))
    assert packed["q"] == be.decide(main) and packed["o"] == be.decide(other)


def _transport(responses, seen):
    it = iter(responses)

    def handler(request: httpx.Request):
        seen.append(request)
        status, body = next(it)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


OK = {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}},
      "usage": {"input_tokens": 42}}


def test_jev_backend_auth_retry_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []
    client = httpx.Client(transport=_transport([(429, {}), (200, OK)], seen))
    be = JevBackend("jev-1.13.0", api_key="test-key", cache_dir=tmp_path, client=client)
    r = Request("s", "q", ["a", "b"])
    assert be.decide(r) == {"a": 0.9, "b": 0.1}
    assert len(seen) == 2 and seen[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(seen[1].content)["model"] == "jev-1.13.0"
    assert be.decide(r) == {"a": 0.9, "b": 0.1}           # cache hit: no third request
    assert len(seen) == 2 and be.stats == {"calls": 1, "cache_hits": 1, "input_tokens": 42}


def test_jev_backend_rejects_other_model(tmp_path):
    bad = dict(OK, model="jev-9.9.9")
    client = httpx.Client(transport=_transport([(200, bad)], []))
    be = JevBackend("jev-1.13.0", api_key="k", cache_dir=tmp_path, client=client)
    with pytest.raises(BackendError):
        be.decide(Request("s", "q", ["a", "b"]))


def test_jev_backend_key_from_either_env(tmp_path, monkeypatch):
    for k in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(BackendError):
        JevBackend("jev-1.13.0", cache_dir=tmp_path)
    monkeypatch.setenv("JEV_API_KEY", "from-env")
    assert JevBackend("jev-1.13.0", cache_dir=tmp_path)._key == "from-env"
    with pytest.raises(ValueError):
        JevBackend("", api_key="k", cache_dir=tmp_path)
