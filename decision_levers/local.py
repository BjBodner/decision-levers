"""Local models and the "one bare call, combined locally" approach.

``combine(bare, vote, lr)`` multiplies the bare model's answer (its prior knowledge) by local evidence from the
labelled pool. In the paper, at 100 labels per class, this plain product trailed the retrieval prompt by 0.7 points
in both fresh-item tests (Runs 4 and 8), and neither showed it matching the prompt; its mean worst-class recall was
29.7% against 42.8% for the prompt (secondary). Prefer the prompt unless privacy rules it out: the product sends
only the item to the vendor, no pool text. Check per-class recall on your own data.

``tempered_combine`` / ``combine(..., weights=...)`` raise each input to an exponent; ``fit_exponents`` fits them by
cross-validation on your own held-out labelled items.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

import numpy as np

from .cascade import _folds
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


# The paper's frozen exponents for the tempered product at about 100 labels per class: alpha (bare), beta (vote),
# gamma (tuned LR). Each is the median of 60 cross-fitted fits (12 dataset rows x 5 cluster folds, gold log-loss on
# Jev 1.13 answers, the dense prior-corrected vote and LR); frozen for the paper's fourth run. Rounded: 0.358 /
# 0.131 / 0.620. Observed: there is no frozen set at 5 per class, and they were fitted with the paper's dense
# features, so refit them on your own held-out data (``fit_exponents``) before relying on them.
PAPER_EXPONENTS = (0.3579, 0.1310, 0.6201)


def combine(bare: Mapping[str, float], *experts: Mapping[str, float], eps: float = 1e-3,
            weights: Sequence[float] | str | None = None) -> Probs:
    """Product of experts: ``(bare + eps) * (expert_1 + eps) * ...``, renormalised over the bare answer's options.
    ``combine(bare, vote, lr)`` is the paper's "bare x vote x tuned LR" (one bare call, no example text sent), the
    plain product of its label-budget tests (G5). It passed on Run 2's items only with items resampled, and was not
    shown to match the prompt on fresh items (-0.7 in Runs 4 and 8).

    ``weights=(a, b, c, ...)``, one per input (bare first), gives the TEMPERED product
    ``(bare + eps)^a * (expert_1 + eps)^b * ...``, renormalised; ``weights="paper"`` uses ``PAPER_EXPONENTS`` and
    needs exactly two experts (vote, LR). ``None`` (default) is the plain product, i.e. all weights 1."""
    if isinstance(weights, str):
        if weights != "paper":
            raise ValueError("weights must be a sequence of numbers, 'paper' or None")
        if len(experts) != 2:
            raise ValueError("weights='paper' is (bare, vote, LR): pass exactly two experts")
        weights = PAPER_EXPONENTS
    w = [1.0] * (1 + len(experts)) if weights is None else [float(x) for x in weights]
    if len(w) != 1 + len(experts):
        raise ValueError(f"need {1 + len(experts)} weights (bare first), got {len(w)}")
    if any(x < 0 or not np.isfinite(x) for x in w):
        raise ValueError("weights must be finite and >= 0")
    opts = list(bare)
    with np.errstate(divide="ignore"):
        z = np.array([w[0] * np.log(float(bare[o]) + eps) for o in opts])
        for wi, ex in zip(w[1:], experts):
            z += np.array([wi * np.log(float(ex.get(o, 0.0)) + eps) for o in opts])
    e = np.exp(z - z.max())
    e /= e.sum()
    return {o: float(v) for o, v in zip(opts, e)}


def tempered_combine(bare: Mapping[str, float], vote: Mapping[str, float], lr: Mapping[str, float],
                     weights: Sequence[float] = PAPER_EXPONENTS, eps: float = 1e-3) -> Probs:
    """``bare^a * vote^b * LR^c`` (each + ``eps``), renormalised; defaults to the paper's frozen exponents
    (``PAPER_EXPONENTS``, about 100 labels per class). Observed: in the paper's cross-fitted analysis it closed the
    gap to the prompt on Banking77 (+0.2) and Run 2 ToxicChat (-0.4) but not on CLINC150 (-0.9); with the frozen
    exponents on Run 4's fresh items it was -0.4 [-1.2, +0.3] (secondary)."""
    return combine(bare, vote, lr, eps=eps, weights=weights)


# ------------------------------------------------------------------------------------------ fitting exponents

def _log_stack(bare: Sequence[Mapping[str, float]], experts: Sequence[Sequence[Mapping[str, float]]],
               options: list[str], eps: float) -> np.ndarray:
    """(n_inputs, n_items, n_options) array of log(p + eps)."""
    cols = [bare, *experts]
    return np.stack([np.log(np.array([[float(p.get(o, 0.0)) for o in options] for p in col]) + eps)
                     for col in cols])


def _fit_w(L: np.ndarray, y: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Non-negative exponents minimising the mean gold log-loss of softmax(sum_j w_j L_j)."""
    from scipy.optimize import minimize

    rows = np.arange(len(y))

    def f(w):
        z = np.tensordot(w, L, axes=1)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        p = np.exp(z - lse[:, None])
        nll = float(-(z[rows, y] - lse).mean())
        grad = -np.array([(Lj[rows, y] - (p * Lj).sum(1)).mean() for Lj in L])
        return nll, grad

    res = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(w0))
    return np.asarray(res.x, dtype=float)


def _scores(L: np.ndarray, w: np.ndarray, y: np.ndarray) -> dict:
    z = np.tensordot(w, L, axes=1)
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return {"accuracy": float((lp.argmax(1) == y).mean()), "log_loss": float(-lp[np.arange(len(y)), y].mean())}


def fit_exponents(bare: Sequence[Mapping[str, float]], experts: Sequence[Sequence[Mapping[str, float]]],
                  gold: Sequence[str], cv: int = 5, groups: Sequence | None = None, seed: int = 0,
                  eps: float = 1e-3) -> dict:
    """Fit tempered-product exponents on YOUR held-out labelled items, by cross-validation.

    ``bare[i]`` is the bare answer for item i; ``experts`` is one list per local model (e.g. ``[votes, lrs]``),
    aligned with ``bare``. The local models must not have been trained on these items. Exponents >= 0 are fitted by
    gold log-loss (the paper's A2 method): on k-1 folds, applied to the k-th (whole ``groups`` per fold), then once
    on all items. Returns ``{"weights", "fold_weights", "crossfit", "plain", "n"}``: deploy ``weights``; compare
    ``crossfit`` (out-of-fold accuracy and log-loss of the tempered product) with ``plain`` (all weights 1, same
    items) before switching. This is a supervised stacker on your labels; it is not evidence about the model."""
    options = list(bare[0]) if bare else []
    if any(list(p) != options for p in bare):
        raise ValueError("every bare answer must have the same options in the same order")
    missing = sorted({g for g in gold if g not in options})
    if missing:
        raise ValueError(f"gold labels not among the options: {missing[:5]}")
    if any(len(col) != len(bare) for col in experts) or len(gold) != len(bare):
        raise ValueError("bare, each expert list and gold must have one entry per item")
    L = _log_stack(bare, experts, options, eps)
    y = np.array([options.index(g) for g in gold])
    w0 = np.ones(L.shape[0])
    folds = _folds(len(y), int(cv), groups, random.Random(f"exponents-cv/{seed}"))
    if len(folds) < 2:
        raise ValueError("cross-validation needs at least 2 non-empty folds")
    fold_w, oof = [], np.zeros(L.shape[1:])
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        w = _fit_w(L[:, tr], y[tr], w0)
        fold_w.append(w.tolist())
        z = np.tensordot(w, L[:, te], axes=1)
        oof[te] = z - z.max(1, keepdims=True)
    lp = oof - np.log(np.exp(oof).sum(1, keepdims=True))
    crossfit = {"accuracy": float((lp.argmax(1) == y).mean()), "log_loss": float(-lp[np.arange(len(y)), y].mean())}
    return {"weights": tuple(_fit_w(L, y, w0).tolist()), "fold_weights": fold_w, "crossfit": crossfit,
            "plain": _scores(L, w0, y), "n": len(y), "options": options}
