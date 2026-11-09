"""decision-levers: information, not instructions, for typed decision models.

Independent project. Not affiliated with, funded by, or endorsed by TypeSafe. 'Jev' and 'TypeSafe' are used only
to name the API this code calls.
"""
from .backends import PAPER_MODEL, Backend, BackendError, JevBackend, MockBackend, ReplayBackend
from .cascade import ClassifierFirstCascade
from .data import load_csv
from .levers import (add_definitions, add_examples, correct_shares, map_back, mean_probs, pack_questions,
                     rename_to_ids, select_examples, state_shares, with_examples)
from .local import PAPER_EXPONENTS, TunedLR, combine, fit_exponents, knn_vote, tempered_combine
from .request import PackedRequest, Request
from .retrieval import DenseRetriever, LabelledPool, TfidfRetriever
from .shortlist import decide_with_shortlist, shortlist

__version__ = "0.2.0"
__all__ = [
    "PAPER_MODEL", "Backend", "BackendError", "JevBackend", "MockBackend", "ReplayBackend",
    "ClassifierFirstCascade", "load_csv", "add_definitions", "add_examples", "correct_shares", "map_back",
    "mean_probs", "pack_questions", "rename_to_ids", "select_examples", "state_shares", "with_examples",
    "PAPER_EXPONENTS", "TunedLR", "combine", "fit_exponents", "knn_vote", "tempered_combine", "PackedRequest",
    "Request", "DenseRetriever", "LabelledPool", "TfidfRetriever", "decide_with_shortlist", "shortlist",
]
