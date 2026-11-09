"""decision-levers: information, not instructions, for typed decision models.

Independent project. Not affiliated with, funded by, or endorsed by TypeSafe. 'Jev' and 'TypeSafe' are used only
to name the API this code calls.
"""
from .backends import PAPER_MODEL, Backend, BackendError, JevBackend, MockBackend
from .cascade import ClassifierFirstCascade
from .levers import (add_definitions, add_examples, correct_shares, map_back, mean_probs, pack_questions,
                     rename_to_ids, select_examples, state_shares, with_examples)
from .local import TunedLR, combine, knn_vote
from .request import PackedRequest, Request
from .retrieval import DenseRetriever, LabelledPool, TfidfRetriever
from .shortlist import decide_with_shortlist, shortlist

__version__ = "0.1.0"
__all__ = [
    "PAPER_MODEL", "Backend", "BackendError", "JevBackend", "MockBackend", "ClassifierFirstCascade",
    "add_definitions", "add_examples", "correct_shares", "map_back", "mean_probs", "pack_questions",
    "rename_to_ids", "select_examples", "state_shares", "with_examples", "TunedLR", "combine", "knn_vote",
    "PackedRequest", "Request", "DenseRetriever", "LabelledPool", "TfidfRetriever", "decide_with_shortlist",
    "shortlist",
]
