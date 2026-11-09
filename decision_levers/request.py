"""Requests to a typed decision model, and their wire format.

A request is one question about one state: ``state`` (the text to decide on), ``question`` (the instruction) and,
for a ``choice`` question, an ordered mapping of option names to optional descriptions. A ``noul`` question is a
yes/no question with no options; its answer comes back as ``{"true": p, "false": 1 - p}``.

The wire body is the one the paper sent to ``POST /v1/systemone``::

    {"model": "<pinned id>", "state": "...",
     "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"Name": "description" | null}}}}

Option ORDER is part of the request (dicts keep insertion order), so two permutations are two different requests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

QTYPES = ("choice", "noul")
Options = Mapping[str, "str | None"]


def _as_options(options: Iterable[str] | Options | None) -> dict[str, str | None] | None:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return {str(k): (None if v is None else str(v)) for k, v in options.items()}
    return {str(k): None for k in options}


@dataclass(frozen=True, eq=True)
class Request:
    """One question about one state.

    ``options`` may be a list of names (no descriptions) or a mapping ``name -> description | None``.
    ``qtype`` defaults to ``"choice"`` when options are given and ``"noul"`` (yes/no) otherwise.
    """

    state: str
    question: str
    options: dict[str, str | None] | None = None
    qtype: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _as_options(self.options))
        qt = self.qtype or ("choice" if self.options else "noul")
        if qt not in QTYPES:
            raise ValueError(f"qtype must be one of {QTYPES}, got {qt!r}")
        if qt == "choice" and not self.options:
            raise ValueError("a choice question needs at least one option")
        if qt == "noul" and self.options:
            raise ValueError("a noul (yes/no) question takes no options")
        object.__setattr__(self, "qtype", qt)

    __hash__ = None  # type: ignore[assignment]  # options is a dict

    # ------------------------------------------------------------------ views
    def option_names(self) -> list[str]:
        """Names the answer is keyed by (``["true", "false"]`` for a yes/no question)."""
        return list(self.options) if self.qtype == "choice" else ["true", "false"]

    def question_dict(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": self.qtype, "instructions": self.question}
        if self.qtype == "choice":
            q["criteria"] = dict(self.options or {})
        return q

    def to_wire(self, model: str) -> dict[str, Any]:
        """The JSON body for one request with the single question named ``"q"``."""
        return {"model": model, "state": self.state, "questions": {"q": self.question_dict()}}

    # ------------------------------------------------------------------ copies
    def with_question(self, question: str) -> "Request":
        return replace(self, question=question)

    def with_options(self, options: Iterable[str] | Options) -> "Request":
        return replace(self, options=_as_options(options))


@dataclass(frozen=True, eq=True)
class PackedRequest:
    """Several questions about ONE state in one request (see ``levers.pack_questions``).

    ``questions`` maps a question name to a ``Request`` with that same state; insertion order is wire order.
    """

    state: str
    questions: dict[str, Request] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("a packed request needs at least one question")
        bad = [n for n, r in self.questions.items() if r.state != self.state]
        if bad:
            raise ValueError(f"questions {bad} are about a different state; packing is for one state only")

    def to_wire(self, model: str) -> dict[str, Any]:
        return {"model": model, "state": self.state,
                "questions": {n: r.question_dict() for n, r in self.questions.items()}}


def parse_answer(answer: Mapping[str, Any]) -> dict[str, float]:
    """One entry of the response's ``answers`` object -> ``{option: probability}``."""
    t = answer.get("type")
    if t == "choice":
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    if t == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    raise ValueError(f"unsupported answer type {t!r}")
