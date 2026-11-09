"""Local models and the "one bare call, combined locally" approach.

``combine(bare, vote, lr)`` multiplies the bare model's answer (its prior knowledge) by local evidence from the
labelled pool. In the paper this product was non-inferior to the retrieval prompt at 100 labels per class on
topical text, while sending no pool text to the vendor. It trailed on short-intent and moderation sets.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .retrieval import LabelledPool, Retriever, as_retriever

Probs = dict[str, float]
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
T_GRID = np.exp(np.linspace(np.log(0.02), np.log(50), 121))


def _prior_weights(options: Sequence[str], prior: Mapping[str, float] | None,
                   pool_share: Mapping[str, float]) -> np.ndarray | None:
    if prior is None:
        return None
    return np.array([prior.get(o, 0.0) / max(pool_share.get(o, 0.0), 1e-6) for o in options])


def knn_vote(text: str, pool: LabelledPool | Retriever, options: Sequence[str], k: int = 10, eps: float = 1e-3,
             prior: Mapping[str, float] | None = None) -> Probs:
    """Similarity-weighted vote of the ``k`` nearest pool rows (weight ``max(cos, 0) + 1e-6/(rank+1)``, ``eps``
    smoothing). With ``prior`` (deployment class shares) the vote is prior-corrected:
    ``p(c) * prior(c) / pool_share(c)``. The paper's frozen vote: dense, k=10, prior-corrected."""
    r = as_retriever(pool)
    sims = r.sims(text)
    out = {o: eps for o in options}
    for rank, i in enumerate(np.argsort(-sims, kind="stable")[:k]):
        lab = r.pool.labels[i]
        out[lab] = out.get(lab, eps) + max(float(sims[i]), 0.0) + 1e-6 / (rank + 1)
    z = sum(out.values())
    p = {o: out[o] / z for o in options}
    w = _prior_weights(options, prior, r.pool.shares())
    if w is not None:
        q = {o: p[o] * wi for o, wi in zip(options, w)}
        zq = sum(q.values())
        p = {o: (v / zq if zq > 0 else 1.0 / len(options)) for o, v in q.items()}
    return p


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _logits(m, X) -> np.ndarray:
    z = m.decision_function(X)
    return np.stack([-z / 2, z / 2], 1) if z.ndim == 1 else z


class TunedLR:
    """Logistic regression on the retriever's features, tuned as in the paper: ``C`` chosen by stratified CV
    log-loss on the pool, temperature fitted on the out-of-fold logits, optional prior correction at predict time.
    The paper used dense (bge-small) features; pass a ``DenseRetriever`` to match it."""

    def __init__(self, pool: LabelledPool | Retriever, options: Sequence[str]):
        self.r, self.options = as_retriever(pool), list(options)
        self.C, self.T, self.model = 10.0, 1.0, None

    def fit(self) -> "TunedLR":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        X = self.r.encode(self.r.pool.texts)
        y = np.array([self.options.index(lab) for lab in self.r.pool.labels])
        self.present = np.unique(y)
        yi = np.searchsorted(self.present, y)
        counts = np.bincount(yi)
        k = int(min(5, counts.min())) if len(counts) else 0
        lr = lambda C: LogisticRegression(C=C, max_iter=3000)  # noqa: E731
        if k >= 2 and len(self.present) >= 2:
            folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(X, yi))
            best = None
            for C in C_GRID:
                oof = np.zeros((len(yi), len(self.present)))
                for tr, te in folds:
                    m = lr(C).fit(X[tr], yi[tr])
                    oof[np.ix_(te, m.classes_)] = _logits(m, X[te])
                nll = -np.log(_softmax(oof)[np.arange(len(yi)), yi].clip(1e-12)).mean()
                if best is None or nll < best[0]:
                    best = (nll, C, oof)
            _, self.C, oof = best
            nlls = [-np.log(_softmax(oof / T)[np.arange(len(yi)), yi].clip(1e-12)).mean() for T in T_GRID]
            self.T = float(T_GRID[int(np.argmin(nlls))])
        self.model = lr(self.C).fit(X, yi) if len(self.present) >= 2 else None
        return self

    def predict(self, texts: Sequence[str], prior: Mapping[str, float] | None = None) -> list[Probs]:
        """Probabilities per text over ``options`` (0 for labels absent from the pool)."""
        if self.model is None and not hasattr(self, "present"):
            self.fit()
        out = np.zeros((len(texts), len(self.options)))
        if self.model is None:
            out[:, self.present[0]] = 1.0
        else:
            out[:, self.present] = _softmax(_logits(self.model, self.r.encode(texts)) / self.T)
        w = _prior_weights(self.options, prior, self.r.pool.shares())
        if w is not None:
            out = out * w[None, :]
            out = out / np.maximum(out.sum(1, keepdims=True), 1e-12)
        return [dict(zip(self.options, row.tolist())) for row in out]


def combine(bare: Mapping[str, float], *experts: Mapping[str, float], eps: float = 1e-3) -> Probs:
    """Product of experts: ``(bare + eps) * (expert_1 + eps) * ...``, renormalised over the bare answer's options.
    ``combine(bare, vote, lr)`` is the paper's "bare x vote x tuned LR" (one bare call, no example text sent)."""
    out = {o: float(p) + eps for o, p in bare.items()}
    for ex in experts:
        for o in out:
            out[o] *= float(ex.get(o, 0.0)) + eps
    z = sum(out.values())
    return {o: v / z for o, v in out.items()}
