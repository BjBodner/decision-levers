"""Load your own labelled data.

``load_csv(path, text_col, label_col)`` returns ``(text, label)`` pairs, the shape ``LabelledPool.from_pairs`` and
``evaluate.holdout_split`` work with. ``LabelledPool.from_csv`` is the one-line version for a pool.

A pandas DataFrame needs no loader: ``LabelledPool.from_pairs(zip(df[text_col], df[label_col]))``.

This folder also holds the toy dataset the examples use (``toy_tickets.csv``, see ``decision_levers.toy``).
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_csv(path: str | Path, text_col: str = "text", label_col: str = "label", *, where: dict | None = None,
             delimiter: str | None = None, encoding: str = "utf-8-sig", skip_empty: bool = True
             ) -> list[tuple[str, str]]:
    """``(text, label)`` pairs from a CSV (or TSV) file with a header row, in file order.

    ``where={"split": "train"}`` keeps only rows whose columns equal those values. Surrounding whitespace is
    stripped; rows with an empty text or label are skipped (``skip_empty=False`` raises instead). The delimiter is
    a tab for ``.tsv``/``.tab`` files and a comma otherwise unless given. ``utf-8-sig`` also reads files with a
    byte-order mark, as Excel writes them."""
    p = Path(path)
    if delimiter is None:
        delimiter = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    with p.open(encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        cols = reader.fieldnames or []
        need = [text_col, label_col, *(where or {})]
        missing = [c for c in need if c not in cols]
        if missing:
            raise KeyError(f"{p.name}: no column {missing}; the header has {cols}")
        out = []
        for n, row in enumerate(reader, start=2):                  # line 1 is the header
            if where and any((row.get(k) or "").strip() != str(v) for k, v in where.items()):
                continue
            text, label = (row.get(text_col) or "").strip(), (row.get(label_col) or "").strip()
            if not text or not label:
                if skip_empty:
                    continue
                raise ValueError(f"{p.name}, line {n}: empty {'text' if not text else 'label'}")
            out.append((text, label))
    return out


__all__ = ["load_csv"]
