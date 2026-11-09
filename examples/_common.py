"""Shared plumbing for the examples: backend choice and a few printing helpers.

Every example runs offline with the mock backend by default. ``--jev`` calls the real API (paid, needs
TYPESAFE_API_KEY); responses are cached on disk, so a rerun costs nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from a source checkout

from decision_levers import PAPER_MODEL, JevBackend, MockBackend, toy  # noqa: E402
from decision_levers.evaluate import hit  # noqa: E402


def backend_and_args(description: str, add_args=None):
    """``(backend, args)``; ``add_args(parser)`` adds an example's own options."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="offline fake backend (default)")
    g.add_argument("--jev", action="store_true", help="the real API (paid; needs TYPESAFE_API_KEY)")
    ap.add_argument("--model", default=PAPER_MODEL, help=f"pinned model id for --jev (default {PAPER_MODEL})")
    if add_args is not None:
        add_args(ap)
    args = ap.parse_args()
    if args.jev:
        print(f"backend: TypeSafe Jev, model {args.model} (paid; cached on disk)")
        return JevBackend(args.model), args
    print("backend: offline mock (numbers illustrate the mechanics only, they are not evidence)")
    return MockBackend(knowledge=toy.MOCK_KNOWLEDGE), args


def backend_from_args(description: str):
    return backend_and_args(description)[0]


def accuracy(answers, gold) -> float:
    return 100.0 * sum(hit(p, g) for p, g in zip(answers, gold)) / max(len(gold), 1)


def show(name: str, answers, gold, tokens: float | None = None) -> None:
    tok = f"   mean input tokens {tokens:,.0f}" if tokens is not None else ""
    print(f"  {name:<34} accuracy {accuracy(answers, gold):5.1f}%{tok}")


def tokens_per_call(backend, before: int, n: int) -> float:
    return (backend.stats["input_tokens"] - before) / max(n, 1)
