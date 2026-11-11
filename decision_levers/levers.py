"""Information levers: edits that ADD information the model lacks. Plus, at the bottom, phrasing edits that add
none and exist only to show that they do nothing (see docs/what_not_to_do.md).

Every function returns a new ``Request``; nothing is mutated. Texts are byte-identical to the paper's arms.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

from .request import PackedRequest, Request
from .retrieval import LabelledPool, Retriever, as_retriever, nearest

SHARE_PREFIX = "Approximate share of past cases by option: "


# ---------------------------------------------------------------------------------------------- DO: examples

def default_per_label(n_options: int) -> int:
    """The paper's cap on examples shown per label: 3, or 10 for a two-option question."""
    return 10 if n_options == 2 else 3


def select_examples(text: str, pool: LabelledPool | Retriever, k: int = 20,
                    per_label: int | None = None, n_options: int = 0) -> list[tuple[str, str]]:
    """``(label, text)`` of the examples to show: the ``k`` nearest pool rows, at most ``per_label`` per label,
    in rank order (nearest first)."""
    r = as_retriever(pool)
    cap = per_label if per_label is not None else default_per_label(n_options)
    shown, count = [], {}
    for i in nearest(r, text, k):
        lab = r.pool.labels[i]
        if count.get(lab, 0) < cap:
            count[lab] = count.get(lab, 0) + 1
            shown.append((lab, r.pool.texts[i]))
    return shown


def with_examples(request: Request, shown: Sequence[tuple[str, str]]) -> Request:
    """Put examples in the option descriptions: ``Examples: "t1"; "t2"``. An option that receives examples has
    its description REPLACED (as in the paper); options without examples keep theirs. Labels that are not options
    of the request are dropped."""
    ex: dict[str, list[str]] = {}
    for lab, t in shown:
        ex.setdefault(lab, []).append(t)
    crit = {o: ("Examples: " + "; ".join(f'"{e}"' for e in ex[o])) if ex.get(o) else d
            for o, d in (request.options or {}).items()}
    return request.with_options(crit)


def add_examples(request: Request, pool: LabelledPool | Retriever, k: int = 20,
                 per_label: int | None = None) -> Request:
    """Retrieved labelled neighbours shown under their labels: the paper's kNN-20 prompt.

    ``per_label=None`` uses the paper's rule (3 per label, 10 for two options). This sends pool texts to the
    vendor; ``local.combine`` gets most of the value at high label budgets without doing so.
    """
    n = len(request.options or {})
    return with_examples(request, select_examples(request.state, pool, k, per_label, n))


# ---------------------------------------------------------------------------------------------- DO: definitions

def add_definitions(request: Request, defs: Mapping[str, str]) -> Request:
    """One-line definitions as option descriptions. Options without a definition keep their description."""
    return request.with_options({o: (defs.get(o) or d) for o, d in (request.options or {}).items()})


def rename_to_ids(request: Request, prefix: str = "option_") -> tuple[Request, dict[str, str]]:
    """Replace option names by neutral ids (``option_01``.. in alphabetical order of the names), keeping the
    request's option order and descriptions. Returns the new request and ``id -> original name``. Ids WITHOUT
    definitions collapse accuracy (2.1% on the paper's invented catalogue). With a one-line description ids did as
    well as names in the paper's names experiment, but were 2.3 points below names on fresh items (not shown within
    2), so readable names with descriptions stay the default."""
    names = sorted(request.options or {})
    to_id = {n: f"{prefix}{i + 1:02d}" for i, n in enumerate(names)}
    req = request.with_options({to_id[o]: d for o, d in (request.options or {}).items()})
    return req, {v: k for k, v in to_id.items()}


def map_back(probs: Mapping[str, float], id_to_name: Mapping[str, str]) -> dict[str, float]:
    return {id_to_name.get(k, k): v for k, v in probs.items()}


# ---------------------------------------------------------------------------------------------- DO: class shares

def share_sentence(shares: Mapping[str, float], options: Sequence[str]) -> str:
    """``Approximate share of past cases by option: A 40%, B 35%, ...`` (decreasing share, whole percent)."""
    top = sorted(((o, float(shares.get(o, 0.0))) for o in options), key=lambda kv: -kv[1])
    return SHARE_PREFIX + ", ".join(f"{o} {100 * v:.0f}%" for o, v in top) + "."


def state_shares(request: Request, shares: Mapping[str, float]) -> Request:
    """Append the TRUE class shares to the instruction. Wrong (shuffled, stale) shares cost accuracy; prefer
    ``correct_shares`` after the call, which was at least as good and adds no tokens."""
    return request.with_question(request.question + " " + share_sentence(shares, request.option_names()))


def correct_shares(probs: Mapping[str, float], shares: Mapping[str, float],
                   train_shares: Mapping[str, float] | None = None) -> dict[str, float]:
    """Post-hoc prior correction ``p'(c) ~ p(c) * shares(c) / train_shares(c)``, renormalised.

    * ``train_shares=None``: multiply the bare answer by the shares (the paper's offline correction).
    * ``train_shares=mean_probs(bare answers)``: contextual calibration (divide out the model's own average).
    * ``train_shares=pool.shares()``: prior shift for a classifier trained on a balanced pool.
    """
    q = {}
    for o, p in probs.items():
        w = float(shares.get(o, 0.0))
        if train_shares is not None:
            w /= max(float(train_shares.get(o, 0.0)), 1e-6)
        q[o] = float(p) * w
    z = sum(q.values())
    return {o: v / z for o, v in q.items()} if z > 0 else {o: 1.0 / len(q) for o in q}


def mean_probs(answers: Sequence[Mapping[str, float]], floor: float = 1e-4) -> dict[str, float]:
    """Average answer vector over unlabelled items (the contextual-calibration bias estimate)."""
    opts = list(answers[0])
    return {o: max(sum(a.get(o, 0.0) for a in answers) / len(answers), floor) for o in opts}


# ---------------------------------------------------------------------------------------------- DO: packing

def pack_questions(main: Request, others: Mapping[str, Request], main_last: bool = True,
                   main_name: str = "q") -> PackedRequest:
    """Several questions about ONE state in one request; with seven unrelated questions the main answer stayed
    within +-2 points of asking it alone (fresh items, Run 8), and packing three related ones saved 51% of input
    tokens. The paper's rule puts the main question LAST."""
    qs = [(n, r) for n, r in others.items()]
    order = qs + [(main_name, main)] if main_last else [(main_name, main)] + qs
    return PackedRequest(main.state, dict(order))


# ---------------------------------------------------------------------------------------------- DON'T: phrasing
# These add no information. Tested in the paper, each stayed within +-2 accuracy points of the bare request.
# They are here so that you can reproduce the null result on your own data, not to be used.

PERSONA = "You are a senior expert analyst with twenty years of experience."
EMPHASIS = "IMPORTANT: THINK CAREFULLY AND ANSWER CORRECTLY."
POLITENESS = "Please, if you would be so kind, help me with this question. Thank you."


def add_persona(request: Request, persona: str = PERSONA) -> Request:
    """DON'T: a persona prefix. Within +-2 points of the bare request (confirmed); a gain of about 1 point is not
    ruled out."""
    return request.with_question(persona + " " + request.question)


def add_emphasis(request: Request, prefix: str = EMPHASIS) -> Request:
    """DON'T: ALL-CAPS emphasis. Within +-2 points of the bare request on fresh items (Run 8)."""
    return request.with_question(prefix + " " + request.question)


def add_politeness(request: Request, prefix: str = POLITENESS) -> Request:
    """DON'T: a politeness prefix. No gain; cost about a point under one wording."""
    return request.with_question(prefix + " " + request.question)


def relabel_top(shown: Sequence[tuple[str, str]], options: Sequence[str], n: int = 3,
                seed: str = "") -> list[tuple[str, str]]:
    """DON'T (stress test only): give the ``n`` nearest shown examples a wrong label, drawn uniformly from the
    other options. Reproduces the paper's wrong-label test, which cost 13 to 34 accuracy points."""
    out = list(shown)
    for r in range(min(n, len(out))):
        lab, t = out[r]
        out[r] = (random.Random(f"{seed}/{r}/{t}").choice([o for o in options if o != lab]), t)
    return out
