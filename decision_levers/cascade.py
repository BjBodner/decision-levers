"""Classifier-first cascade: a local model answers the items it is confident about; the rest go to the model.

The threshold is fitted on HELD-OUT labelled items where you have both the local answer and the full model
answer. Two targets:

* ``max_acc_loss``: the smallest share sent to the model whose accuracy stays within ``max_acc_loss`` of sending
  every item (the paper's rule used 0.005, i.e. half a point);
* ``min_recall`` + ``positive``: the smallest share sent that keeps the recall of a rare class at or above a floor.
  Use this one for rare positives: an accuracy-chosen threshold kept accuracy on a moderation set but cost 7.3
  points of precision and most of the ranking quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

Probs = Mapping[str, float]


def _argmax(p: Probs) -> str:
    return max(p, key=p.get)


@dataclass
class ClassifierFirstCascade:
    """Route an item to the model iff the local model's top probability is at or below ``threshold``."""

    threshold: float = float("inf")
    report: dict = field(default_factory=dict)

    def fit(self, local: Sequence[Probs], model: Sequence[Probs], gold: Sequence[str],
            max_acc_loss: float | None = 0.005, min_recall: float | None = None,
            positive: str | None = None) -> "ClassifierFirstCascade":
        """Choose the threshold on held-out items. ``local[i]`` / ``model[i]`` are the two answers for item i."""
        if min_recall is not None and positive is None:
            raise ValueError("min_recall needs the positive class")
        conf = np.array([max(p.values()) for p in local])
        loc_pred = np.array([_argmax(p) for p in local])
        mod_pred = np.array([_argmax(p) for p in model])
        g = np.asarray(gold)
        full_acc = float((mod_pred == g).mean())
        grid = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()) | {1.0 + 1e-9, -1.0})
        feasible = []
        for th in grid:
            loc = conf > th
            pred = np.where(loc, loc_pred, mod_pred)
            acc, sent = float((pred == g).mean()), float(1 - loc.mean())
            if min_recall is not None:
                pos = g == positive
                rec = float((pred[pos] == positive).mean()) if pos.any() else 1.0
                ok = rec >= min_recall
            else:
                ok = acc >= full_acc - float(max_acc_loss or 0.0)
            if ok:
                feasible.append((sent, -acc, th))
        if feasible:
            sent, neg_acc, th = min(feasible)
            self.threshold = float(th)
        else:                                            # nothing meets the target: send everything
            sent, neg_acc, self.threshold = 1.0, -full_acc, float("inf")
        self.report = {"share_sent": sent, "accuracy": -neg_acc, "full_model_accuracy": full_acc,
                       "local_accuracy": float((loc_pred == g).mean()), "n": len(g), "feasible": bool(feasible)}
        return self

    def route(self, local: Sequence[Probs]) -> np.ndarray:
        """True where the item must go to the model."""
        return np.array([max(p.values()) <= self.threshold for p in local], dtype=bool)

    def predict(self, texts: Sequence[str], local_fn: Callable[[Sequence[str]], list],
                model_fn: Callable[[str], Probs]) -> tuple[list[dict], float]:
        """Answers for ``texts`` and the share sent to the model. ``model_fn`` is called only for routed items."""
        local = local_fn(texts)
        send = self.route(local)
        out = [dict(model_fn(t)) if s else dict(p) for t, p, s in zip(texts, local, send)]
        return out, float(send.mean()) if len(send) else 0.0
