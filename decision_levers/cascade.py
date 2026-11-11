"""Classifier-first cascade: a local model answers the items it is confident about; the rest go to the model.

The threshold is fitted on HELD-OUT labelled items where you have both the local answer and the full model
answer. Targets:

* ``max_acc_loss``: the smallest share sent to the model whose accuracy stays within ``max_acc_loss`` of sending
  every item (the paper's rule used 0.005, i.e. half a point);
* ``min_recall`` and/or ``min_precision`` with ``positive``: the smallest share sent that keeps the recall and/or
  the precision of a rare class at or above a floor. Use these for rare positives: an accuracy-chosen threshold kept
  accuracy on a moderation set but cost 7.3 points of precision and most of the ranking quality (observed).

**Thresholds chosen on the same items they are scored on are optimistic.** ``fit`` picks the threshold that just
meets the target on the calibration items, so its own report flatters it (the paper calls these numbers "optimistic
by construction"). Pass ``cv=k`` to cross-fit: the calibration items are split into ``k`` folds (whole near-duplicate
groups per fold when ``groups`` is given); for each fold the threshold is chosen on the other folds and applied to
it. ``report["crossfit"]`` then holds the out-of-fold accuracy, share sent, recall and precision: quote those, not
the in-sample ones. The deployed threshold is still the one chosen on all calibration items.

What leaves your machine: items answered locally send nothing, but every forwarded item is sent to whatever
``model_fn`` calls. With the paper's cascade that is the kNN-20 prompt, so forwarded items go out WITH their
nearest labelled pool texts. A cascade reduces egress; it does not remove it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

Probs = Mapping[str, float]


def _argmax(p: Probs) -> str:
    return max(p, key=p.get)


def _metrics(pred: np.ndarray, sent: np.ndarray, gold: np.ndarray, positive: str | None) -> dict:
    out = {"accuracy": float((pred == gold).mean()) if len(gold) else 0.0,
           "share_sent": float(sent.mean()) if len(sent) else 0.0}
    if positive is not None:
        pos, called = gold == positive, pred == positive
        out["recall"] = float((called[pos]).mean()) if pos.any() else 1.0
        # no positive prediction at all: precision is undefined; 0 so that "never flag" cannot pass a floor
        out["precision"] = float(pos[called].mean()) if called.any() else 0.0
    return out


def _choose(conf: np.ndarray, loc_pred: np.ndarray, mod_pred: np.ndarray, gold: np.ndarray,
            max_acc_loss: float | None, min_recall: float | None, min_precision: float | None,
            positive: str | None) -> tuple[float, bool]:
    """The threshold with the smallest share sent that meets the target on these items (ties: higher accuracy).
    Returns ``(threshold, feasible)``; nothing feasible -> ``inf`` (send every item)."""
    full_acc = float((mod_pred == gold).mean()) if len(gold) else 0.0
    rare = min_recall is not None or min_precision is not None
    grid = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()) | {1.0 + 1e-9, -1.0}) if len(conf) \
        else [1.0 + 1e-9]
    best = None
    for th in grid:
        loc = conf > th
        m = _metrics(np.where(loc, loc_pred, mod_pred), ~loc, gold, positive)
        if rare:
            ok = (min_recall is None or m["recall"] >= min_recall) and \
                 (min_precision is None or m["precision"] >= min_precision)
        else:
            ok = m["accuracy"] >= full_acc - float(max_acc_loss or 0.0)
        if ok:
            key = (m["share_sent"], -m["accuracy"], th)
            if best is None or key < best:
                best = key
    return (float(best[2]), True) if best else (float("inf"), False)


def _folds(n: int, k: int, groups: Sequence[Hashable] | None, rng: random.Random) -> list[np.ndarray]:
    """``k`` folds of item indices; whole groups stay together, folds are balanced by size."""
    g = list(groups) if groups is not None else list(range(n))
    if len(g) != n:
        raise ValueError("groups must have one entry per item")
    members: dict[Hashable, list[int]] = {}
    for i, gi in enumerate(g):
        members.setdefault(gi, []).append(i)
    keys = list(members)
    rng.shuffle(keys)
    keys.sort(key=lambda x: -len(members[x]))           # largest groups first, ties in shuffled order
    folds: list[list[int]] = [[] for _ in range(k)]
    for key in keys:
        min(folds, key=len).extend(members[key])
    return [np.array(sorted(f), dtype=int) for f in folds if f]


@dataclass
class ClassifierFirstCascade:
    """Route an item to the model iff the local model's top probability is at or below ``threshold``."""

    threshold: float = float("inf")
    report: dict = field(default_factory=dict)

    def fit(self, local: Sequence[Probs], model: Sequence[Probs], gold: Sequence[str],
            max_acc_loss: float | None = 0.005, min_recall: float | None = None,
            positive: str | None = None, min_precision: float | None = None,
            cv: int | None = None, groups: Sequence[Hashable] | None = None, repeats: int = 1,
            seed: int = 0) -> "ClassifierFirstCascade":
        """Choose the threshold on held-out items. ``local[i]`` / ``model[i]`` are the two answers for item i.

        Targets: ``max_acc_loss`` (default), or ``min_recall`` and/or ``min_precision`` for the class ``positive``
        (when either is given the accuracy target is not used). ``cv=k`` adds a k-fold cross-fitted estimate in
        ``report["crossfit"]`` (``groups``: near-duplicate group per item, kept within one fold; ``repeats``:
        number of random partitions averaged, the paper used 200 two-fold splits)."""
        if (min_recall is not None or min_precision is not None) and positive is None:
            raise ValueError("min_recall / min_precision need the positive class")
        if not (len(local) == len(model) == len(gold)):
            raise ValueError("local, model and gold must have one entry per item")
        conf = np.array([max(p.values()) for p in local])
        loc_pred = np.array([_argmax(p) for p in local])
        mod_pred = np.array([_argmax(p) for p in model])
        g = np.asarray(gold)
        target = dict(max_acc_loss=max_acc_loss, min_recall=min_recall, min_precision=min_precision,
                      positive=positive)

        self.threshold, feasible = _choose(conf, loc_pred, mod_pred, g, **target)
        loc = conf > self.threshold
        m = _metrics(np.where(loc, loc_pred, mod_pred), ~loc, g, positive)
        self.report = {**m, "full_model_accuracy": float((mod_pred == g).mean()) if len(g) else 0.0,
                       "local_accuracy": float((loc_pred == g).mean()) if len(g) else 0.0,
                       "n": len(g), "feasible": feasible, "optimistic": True}
        if positive is not None:
            self.report["full_model"] = _metrics(mod_pred, np.ones(len(g), bool), g, positive)
        if cv is not None:
            self.report["crossfit"] = self._crossfit(conf, loc_pred, mod_pred, g, int(cv), groups,
                                                     int(repeats), seed, target)
        return self

    @staticmethod
    def _crossfit(conf, loc_pred, mod_pred, g, k, groups, repeats, seed, target) -> dict:
        if k < 2:
            raise ValueError("cv needs at least 2 folds")
        rng = random.Random(f"cascade-cv/{seed}")
        runs, thresholds = [], []
        for _ in range(max(1, repeats)):
            folds = _folds(len(g), k, groups, rng)
            if len(folds) < 2:
                raise ValueError("cross-fitting needs at least 2 non-empty folds (more items or groups)")
            pred, sent = np.empty(len(g), dtype=object), np.zeros(len(g), bool)
            for te in folds:
                tr = np.setdiff1d(np.arange(len(g)), te)
                th, _ = _choose(conf[tr], loc_pred[tr], mod_pred[tr], g[tr], **target)
                thresholds.append(th)
                loc = conf[te] > th
                pred[te] = np.where(loc, loc_pred[te], mod_pred[te])
                sent[te] = ~loc
            runs.append(_metrics(pred.astype(str), sent, g, target["positive"]))
        out = {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
        finite = [t for t in thresholds if np.isfinite(t)]
        out.update({"folds": k, "repeats": max(1, repeats), "fold_thresholds": thresholds,
                    "median_threshold": float(np.median(finite)) if finite else float("inf")})
        full_acc = float((mod_pred == g).mean()) if len(g) else 0.0
        if target["min_recall"] is not None or target["min_precision"] is not None:
            out["meets_target"] = bool((target["min_recall"] is None or out["recall"] >= target["min_recall"])
                                       and (target["min_precision"] is None
                                            or out["precision"] >= target["min_precision"]))
        else:
            out["meets_target"] = bool(out["accuracy"] >= full_acc - float(target["max_acc_loss"] or 0.0))
        return out

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
