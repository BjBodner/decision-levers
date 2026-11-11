"""Confirm before you ship: the evaluation protocol of the paper, in small pieces.

screen (cheap, many arms, with null controls) -> freeze (hash the exact config) -> confirm (held-out,
de-duplicated items; paired cluster bootstrap; equivalence or non-inferiority where you claim "no worse"; Holm).

Why: on a cheap API a plausible screening rule promoted 12 of 53 content-free edits. Null controls (an identical
resend and a content-free edit) tell you how big an "effect" pure noise makes on YOUR data; promote only above it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from .request import Request

Probs = Mapping[str, float]
LOGP_CLIP = 0.005           # Jev often answers exactly 0.00; clip before taking logs
PAPER_FLOOR = 0.04          # the paper's promotion floor in gold log-prob, set above 53 measured null effects


# ---------------------------------------------------------------------------------------------- splits

def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split())


def near_duplicate_groups(texts: Sequence[str], thresh: float = 0.9) -> list[int]:
    """Group id per text: exact duplicates after normalisation, then TF-IDF (word 1-2-gram) cosine >= ``thresh``,
    merged transitively. Ids are contiguous in order of first appearance."""
    keys: dict[str, int] = {}
    uid = [keys.setdefault(_norm(t) or str(t), len(keys)) for t in texts]
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if len(keys) > 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode").fit_transform(list(keys))
        except ValueError:                          # empty vocabulary: exact duplicates only
            X = None
        if X is not None:
            S = (X @ X.T).tocoo()
            for a, b, v in zip(S.row, S.col, S.data):
                if a < b and v >= thresh:
                    ra, rb = find(int(a)), find(int(b))
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    ids: dict[int, int] = {}
    return [ids.setdefault(find(u), len(ids)) for u in uid]


def holdout_split(texts: Sequence[str], test_frac: float = 0.3, seed: int = 0,
                  thresh: float = 0.9) -> tuple[list[int], list[int], list[int]]:
    """``(train_idx, test_idx, groups)`` with every near-duplicate group wholly on one side, so no test item has a
    twin in the pool (twins inflated the paper's development gains by up to 3 points)."""
    groups = near_duplicate_groups(texts, thresh)
    order = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    target, test_g, n = test_frac * len(texts), set(), 0
    size = np.bincount(groups)
    for g in order:
        if n >= target:
            break
        test_g.add(g)
        n += int(size[g])
    test = [i for i, g in enumerate(groups) if g in test_g]
    train = [i for i, g in enumerate(groups) if g not in test_g]
    return train, test, groups


# ---------------------------------------------------------------------------------------------- statistics

def hit(p: Probs, gold: str) -> float:
    return float(max(p, key=p.get) == gold)


def class_metrics(answers: Sequence[Probs], gold: Sequence[str], min_items: int = 1) -> dict:
    """Accuracy, macro-F1 (over the gold classes present), per-class recall and the worst-class recall (over classes
    with at least ``min_items`` gold items; the paper used 10). Accuracy alone hid the costs the paper found on
    small classes: on CFPB the local combination's smallest-class recall trailed the prompt's by 10 to 16 points."""
    pred = [max(p, key=p.get) for p in answers]
    classes = sorted(set(gold))
    recall, f1 = {}, []
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        n_gold, n_pred = sum(g == c for g in gold), sum(p == c for p in pred)
        recall[c] = tp / n_gold
        f1.append(0.0 if tp == 0 else 2 * tp / (n_gold + n_pred))
    eligible = {c: r for c, r in recall.items() if sum(g == c for g in gold) >= min_items}
    worst = min(eligible, key=eligible.get) if eligible else None
    return {"accuracy": float(np.mean([p == g for p, g in zip(pred, gold)])) if gold else 0.0,
            "macro_f1": float(np.mean(f1)) if f1 else 0.0, "recall": recall,
            "worst_class": worst, "worst_class_recall": eligible.get(worst, float("nan")) if worst else float("nan")}


def gold_logp(p: Probs, gold: str, clip: float = LOGP_CLIP) -> float:
    return math.log(max(float(p.get(gold, 0.0)), clip))


def bootstrap_means(values: Sequence[float], clusters: Sequence | None = None, draws: int = 20000,
                    seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean. With ``clusters`` (e.g. near-duplicate groups, or the gold class when
    there are many options) whole clusters are resampled: sum of resampled values / number of resampled items."""
    v = np.asarray(values, float)
    cl = np.arange(len(v)) if clusters is None else np.unique(np.asarray(clusters), return_inverse=True)[1]
    S, N = np.bincount(cl, weights=v), np.bincount(cl).astype(float)
    rng, G, out = np.random.default_rng(seed), len(S), np.empty(draws)
    for s in range(0, draws, 1000):
        e = min(draws, s + 1000)
        W = np.zeros((e - s, G))
        np.add.at(W, (np.repeat(np.arange(e - s), G), rng.integers(0, G, size=(e - s) * G)), 1.0)
        out[s:e] = (W @ S) / np.maximum(W @ N, 1e-12)
    return out


def p_value(kind: str, draws: np.ndarray, margin: float = 0.0) -> float:
    """Bootstrap p for ``sup`` (> 0), ``harm`` (< 0), ``noninf`` (> -margin) or ``tost`` (inside +-margin)."""
    n, lo, hi = len(draws), int((draws <= -margin).sum()), int((draws >= margin).sum())
    if kind == "sup":
        return (1 + int((draws <= 0).sum())) / (n + 1)
    if kind == "harm":
        return (1 + int((draws >= 0).sum())) / (n + 1)
    if kind == "noninf":
        return (1 + lo) / (n + 1)
    if kind == "tost":
        return max(1 + lo, 1 + hi) / (n + 1)
    raise ValueError(kind)


def paired_bootstrap(diffs: Sequence[float], clusters: Sequence | None = None, ci: float = 0.90,
                     draws: int = 20000, seed: int = 0) -> dict:
    """Estimate, ``ci`` interval and one-sided p-values for paired per-item differences (treatment - control)."""
    d = bootstrap_means(diffs, clusters, draws, seed)
    a = (1 - ci) / 2
    return {"estimate": float(np.mean(diffs)), "lo": float(np.quantile(d, a)), "hi": float(np.quantile(d, 1 - a)),
            "p_sup": p_value("sup", d), "p_harm": p_value("harm", d), "n": len(diffs), "draws": d}


def tost(diffs: Sequence[float], margin: float = 0.02, clusters: Sequence | None = None, alpha: float = 0.05,
         draws: int = 20000, seed: int = 0) -> dict:
    """Two one-sided tests: equivalent iff the mean difference is inside ``+-margin`` at level ``alpha`` (the 90%
    interval lies inside the margin). The paper used +-2 accuracy points."""
    r = paired_bootstrap(diffs, clusters, 1 - 2 * alpha, draws, seed)
    p = p_value("tost", r["draws"], margin)
    return {**r, "margin": margin, "p_tost": p, "equivalent": p <= alpha}


def holm(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over a family of tests fixed in advance: adjusted p and reject flag per test."""
    m, out, running, still = len(pvals), {}, 0.0, True
    for i, k in enumerate(sorted(pvals, key=lambda k: (pvals[k], k))):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        still = still and pvals[k] <= alpha / (m - i)
        out[k] = {"p_holm": running, "reject": bool(still)}
    return out


# ---------------------------------------------------------------------------------------------- null controls

def content_free_edit(request: Request, tag: str = "") -> Request:
    """A meaningless reference tag appended to the instruction: changes the bytes, adds no information."""
    n = int(hashlib.sha1(f"ref{tag}{request.state}".encode()).hexdigest()[:8], 16) % 9000 + 1000
    return request.with_question(f"{request.question} [ref {n}]")


def null_controls(backend, requests: Sequence[Request], gold: Sequence[str]) -> dict[str, dict]:
    """Per-item null effects of an identical resend (``rep=1``) and of a content-free edit, in gold log-prob and
    accuracy, plus flip rates. Their size is the noise any real lever must clear."""
    base = backend.decide_many(list(requests))
    arms = {"identical_resend": backend.decide_many(list(requests), rep=1),
            "content_free_edit": backend.decide_many([content_free_edit(r) for r in requests])}
    out = {}
    for name, ans in arms.items():
        dl = [gold_logp(t, g) - gold_logp(c, g) for c, t, g in zip(base, ans, gold)]
        da = [hit(t, g) - hit(c, g) for c, t, g in zip(base, ans, gold)]
        flips = [max(c, key=c.get) != max(t, key=t.get) for c, t in zip(base, ans)]
        out[name] = {"d_logp": float(np.mean(dl)), "d_acc": float(np.mean(da)), "flip": float(np.mean(flips)),
                     "per_item_logp": dl}
    return out


def promotion_floor(null_effects: Sequence[float], step: float = 0.01) -> float:
    """The smallest multiple of ``step`` strictly above the largest absolute null effect (the paper's 0.04 sat
    just above its largest null, 0.0345). Calibrate on nulls you did NOT tune on, or the floor is circular."""
    return (math.floor(max(abs(x) for x in null_effects) / step) + 1) * step


# ---------------------------------------------------------------------------------------------- compare

def compare(control_fn: Callable, treatment_fn: Callable, items: Sequence, gold: Sequence[str],
            clusters: Sequence | None = None, metric: str = "acc", floor: float = 0.0,
            margin: float | None = None, aa_flip: float | None = None, seed: int = 0) -> dict:
    """Paired comparison of two arms on the same items. ``*_fn(item) -> {option: probability}``.

    Decision (screening rule): ``promote`` if the 90% interval excludes 0, the effect is at least ``floor`` and the
    flip rate beats the A/A flip rate; ``harm`` if the interval is below 0 and the effect is at most ``-floor``;
    otherwise ``no effect``. With ``margin`` a TOST equivalence result is added. A screen only nominates: freeze
    the winner, then confirm it on fresh held-out items.
    """
    pc, pt = [control_fn(x) for x in items], [treatment_fn(x) for x in items]
    f = hit if metric == "acc" else gold_logp
    diffs = [f(t, g) - f(c, g) for c, t, g in zip(pc, pt, gold)]
    r = paired_bootstrap(diffs, clusters, seed=seed)
    flip = float(np.mean([max(c, key=c.get) != max(t, key=t.get) for c, t in zip(pc, pt)]))
    lo, hi, est = r["lo"], r["hi"], r["estimate"]
    dec = ("promote" if lo > 0 and est >= floor and (aa_flip is None or flip > aa_flip) else
           "harm" if hi < 0 and est <= -floor else "no effect")
    out = {k: v for k, v in r.items() if k != "draws"}
    out.update({"metric": metric, "flip": flip, "decision": dec,
                "control": float(np.mean([f(c, g) for c, g in zip(pc, gold)])),
                "treatment": float(np.mean([f(t, g) for t, g in zip(pt, gold)]))})
    if margin is not None:
        t = tost(diffs, margin, clusters, seed=seed)
        out.update({"equivalent": t["equivalent"], "p_tost": t["p_tost"], "margin": margin})
    return out


def freeze(config: Mapping) -> str:
    """SHA-256 of a JSON-serialisable config (arms, texts, thresholds, items): record it BEFORE confirming."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
