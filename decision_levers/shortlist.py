"""Labels-only shortlist: restrict the options to the labels of the k nearest pool neighbours, without sending
any example text. If all k neighbours share one label, answer locally and skip the call.

In the paper (exploratory) this kept short-intent sets within about 2.5 points of the kNN-20 prompt at about 5%
of its tokens, with about 80% of items needing no call. It HURT topical text (tweets: below the bare model),
where the gold label was often not among the neighbours' labels.
"""
from __future__ import annotations

from typing import Sequence

from .request import Request
from .retrieval import LabelledPool, Retriever, as_retriever, nearest

Probs = dict[str, float]


def shortlist(text: str, pool: LabelledPool | Retriever, k: int = 5) -> list[str]:
    """Distinct labels of the ``k`` nearest neighbours, in rank order (the paper used dense neighbours, k=5)."""
    r = as_retriever(pool)
    out: list[str] = []
    for i in nearest(r, text, k):
        lab = r.pool.labels[i]
        if lab not in out:
            out.append(lab)
    return out


def restrict(request: Request, labels: Sequence[str]) -> Request:
    """The request with only the shortlisted options, in the request's own option order."""
    keep = set(labels)
    return request.with_options({o: d for o, d in (request.options or {}).items() if o in keep})


def decide_with_shortlist(request: Request, pool: LabelledPool | Retriever, backend, k: int = 5) -> tuple[Probs, bool]:
    """Probabilities over ALL options of ``request`` (0 outside the shortlist) and whether a call was made.
    A one-label shortlist answers ``p = 1`` locally; an empty overlap with the options falls back to the full
    request."""
    opts = list(request.options or {})
    sl = [lab for lab in shortlist(request.state, pool, k) if lab in set(opts)]
    if len(sl) == 1:
        return {o: (1.0 if o == sl[0] else 0.0) for o in opts}, False
    if not sl:
        return dict(backend.decide(request)), True
    p = backend.decide(restrict(request, sl))
    return {o: float(p.get(o, 0.0)) for o in opts}, True
