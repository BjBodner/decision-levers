"""Labelled pools and nearest-neighbour retrieval (TF-IDF by default, dense embeddings optional).

Both retrievers expose ``pool``, ``encode(texts)`` (the feature matrix of any texts) and ``sims(text)`` (cosine of one
query to every pool row), so the levers, the local models and the shortlist work with either.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

DENSE_MODEL = "BAAI/bge-small-en-v1.5"       # the dense encoder the paper used


@dataclass
class LabelledPool:
    """Labelled examples ``(label, text)`` in a fixed order. Build one from training data only, never test items."""

    labels: list[str]
    texts: list[str]
    _retriever: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.texts):
            raise ValueError("labels and texts must have the same length")

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], per_class: int | None = None) -> "LabelledPool":
        """From ``(text, label)`` pairs, keeping the first ``per_class`` of each label (all when None)."""
        labels, texts, seen = [], [], Counter()
        for text, label in pairs:
            if per_class is None or seen[label] < per_class:
                seen[label] += 1
                labels.append(str(label))
                texts.append(str(text))
        return cls(labels, texts)

    @classmethod
    def from_csv(cls, path, text_col: str = "text", label_col: str = "label", per_class: int | None = None,
                 **kwargs) -> "LabelledPool":
        """From a CSV file with a header row (see ``decision_levers.data.load_csv`` for ``where=``, ``delimiter=``
        and the other options). Build the pool from training rows only, never from the items you test on."""
        from .data import load_csv
        return cls.from_pairs(load_csv(path, text_col, label_col, **kwargs), per_class=per_class)

    def __len__(self) -> int:
        return len(self.texts)

    def cap(self, per_class: int) -> "LabelledPool":
        """The first ``per_class`` examples of each label (the paper's label-budget sweep)."""
        return LabelledPool.from_pairs(zip(self.texts, self.labels), per_class)

    def shares(self) -> dict[str, float]:
        """Share of each label in the pool (needed for prior correction of local models)."""
        c = Counter(self.labels)
        return {k: v / len(self.labels) for k, v in c.items()}

    def with_wrong_labels(self, frac: float, options: Sequence[str], seed: int = 1) -> "LabelledPool":
        """A copy with ``frac`` of the labels replaced by another option: a stress test for annotator noise."""
        labels = list(self.labels)
        rng = random.Random(f"wrong/{seed}")
        for i in rng.sample(range(len(labels)), round(frac * len(labels))):
            labels[i] = rng.choice([o for o in options if o != labels[i]])
        return LabelledPool(labels, list(self.texts))

    def retriever(self) -> "TfidfRetriever":
        """The default TF-IDF retriever over this pool (built once, then reused)."""
        if self._retriever is None:
            self._retriever = TfidfRetriever(self)
        return self._retriever  # type: ignore[return-value]


class TfidfRetriever:
    """TF-IDF on word 1-2-grams, sublinear tf, accent stripping, cosine similarity (the paper's default).

    The vectorizer is fitted on the pool texts plus any ``extra_texts`` (the paper also included the unlabelled
    items it scored; that is transductive but label-free).
    """

    def __init__(self, pool: LabelledPool, extra_texts: Sequence[str] = ()):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.pool = pool
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
        self.vec.fit(list(pool.texts) + list(extra_texts))
        self.matrix = self.vec.transform(pool.texts)

    def encode(self, texts: Sequence[str]):
        return self.vec.transform(list(texts))

    def sims(self, text: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([text]).T).toarray().ravel()


class DenseRetriever:
    """Cosine similarity of L2-normalised sentence embeddings. Needs ``pip install decision-levers[dense]``
    (fastembed) unless an ``embed`` callable ``list[str] -> (n, dim) array`` is passed."""

    def __init__(self, pool: LabelledPool, embed: Callable[[list[str]], np.ndarray] | None = None,
                 model: str = DENSE_MODEL):
        if embed is None:
            from fastembed import TextEmbedding           # optional extra
            m = TextEmbedding(model)
            embed = lambda texts: np.array(list(m.embed(texts)), dtype=np.float32)  # noqa: E731
        self._embed, self.pool, self._memo = embed, pool, {}
        self.matrix = self.encode(pool.texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if todo:
            v = np.asarray(self._embed(list(todo)), dtype=np.float32)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            self._memo.update(zip(todo, v))
        if not texts:
            return np.zeros((0, self.matrix.shape[1] if hasattr(self, "matrix") else 1), dtype=np.float32)
        return np.stack([self._memo[t] for t in texts])

    def sims(self, text: str) -> np.ndarray:
        return np.asarray(self.matrix @ self.encode([text])[0]).ravel()


Retriever = TfidfRetriever | DenseRetriever


def as_retriever(pool_or_retriever: LabelledPool | Retriever) -> Retriever:
    return pool_or_retriever.retriever() if isinstance(pool_or_retriever, LabelledPool) else pool_or_retriever


def nearest(retriever: Retriever, text: str, k: int) -> list[int]:
    """Pool indices of the ``k`` most similar rows, most similar first (stable on ties)."""
    return [int(i) for i in np.argsort(-retriever.sims(text), kind="stable")[:k]]
