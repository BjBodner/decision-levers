"""A tiny hand-written helpdesk dataset (72 tickets, 5 classes, imbalanced) for the examples and tests.

Written for this package; no licensed data. Splits: ``train`` (the labelled pool), ``calib`` (held out, for
fitting thresholds) and ``test``. Too small for evidence: use it to see how the pieces fit together.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources

from .request import Request

OPTIONS = ["Billing", "Account access", "Shipping", "Refund", "Bug report"]
QUESTION = "Which team should handle this support ticket?"
DEFINITIONS = {
    "Billing": "Charges, invoices, receipts, fees, payment cards and plan prices.",
    "Account access": "Signing in: passwords, locked accounts, verification codes, login email or username.",
    "Shipping": "Delivery of a parcel or order: tracking, courier, delivery address, late or damaged package.",
    "Refund": "Getting money back for a return or a cancelled order.",
    "Bug report": "The app or website crashes, freezes, shows an error or a page does not load.",
}

# What the offline mock "already knows" about each option name (a stand-in for a pretrained model's prior).
# Deliberately incomplete, so that information in the request still helps.
MOCK_KNOWLEDGE = {
    "Billing": ["invoice", "fee"],
    "Account access": ["password"],
    "Shipping": ["parcel", "tracking"],
    "Refund": ["refund"],
    "Bug report": ["error"],
}


@dataclass(frozen=True)
class Ticket:
    text: str
    label: str
    split: str


def load(split: str | None = None) -> list[Ticket]:
    """All tickets, or those of one split (``train``, ``calib``, ``test``)."""
    with resources.files("decision_levers").joinpath("data/toy_tickets.csv").open(encoding="utf-8") as f:
        rows = [Ticket(r["text"], r["label"], r["split"]) for r in csv.DictReader(f)]
    return [r for r in rows if split is None or r.split == split]


def bare_request(text: str) -> Request:
    """The bare request: option names only, no descriptions."""
    return Request(state=text, question=QUESTION, options=OPTIONS)


def train_shares() -> dict[str, float]:
    tr = load("train")
    return {o: sum(t.label == o for t in tr) / len(tr) for o in OPTIONS}
