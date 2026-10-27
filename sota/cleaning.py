"""Cleaning rules: reusable, exportable as JSON.

Term syntax (case-insensitive, matched on whole words like IEEE Xplore):
  plasma        -> the word "plasma"
  cardio*       -> words starting with "cardio"
  blood vessel  -> the exact phrase
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

RULE_FIELDS = ["document_title", "abstract", "ieee_terms", "author_keywords", "mesh_terms"]


@dataclass
class Rules:
    name: str = "reglas"
    exclude_terms: list[str] = field(default_factory=list)  # drop docs matching any
    require_terms: list[str] = field(default_factory=list)  # if set, keep only docs matching any
    exclude_ids: list[str] = field(default_factory=list)    # manually removed docs
    keep_ids: list[str] = field(default_factory=list)       # manually kept docs (override terms)
    synonyms: dict[str, str] = field(default_factory=dict)  # applied before vectorizing
    min_year: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str | bytes) -> "Rules":
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def add(self, attr: str, values) -> int:
        cur = getattr(self, attr)
        new = [v for v in dict.fromkeys(values) if v and v not in cur]
        cur.extend(new)
        return len(new)


def _pattern(terms: list[str]) -> re.Pattern | None:
    parts = []
    for t in terms:
        t = t.strip().lower()
        if t:
            parts.append(r"\b" + re.escape(t).replace(r"\*", r"\w*").replace(r"\ ", r"\s+") + r"\b")
    return re.compile("(" + "|".join(parts) + ")", re.I) if parts else None


def haystack(df: pd.DataFrame) -> pd.Series:
    return df[RULE_FIELDS].fillna("").astype(str).agg(" | ".join, axis=1)


def matches(df: pd.DataFrame, terms: list[str], text: pd.Series | None = None) -> pd.Series:
    """First matched term per row (NaN if none)."""
    pat = _pattern(terms)
    if pat is None:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    text = haystack(df) if text is None else text
    return text.str.extract(pat, expand=False).str.lower()


def apply(df: pd.DataFrame, rules: Rules, text: pd.Series | None = None) -> pd.Series:
    """Removal reason per row ('' = kept)."""
    text = haystack(df) if text is None else text
    reason = pd.Series("", index=df.index, dtype="object")
    if rules.require_terms:
        reason[matches(df, rules.require_terms, text).isna()] = "sin términos requeridos"
    if rules.min_year:
        reason[df["publication_year"].fillna(0) < rules.min_year] = f"año < {rules.min_year}"
    hit = matches(df, rules.exclude_terms, text)
    reason[hit.notna()] = "término: " + hit[hit.notna()]
    reason[df["id"].isin(rules.exclude_ids)] = "manual"
    reason[df["id"].isin(rules.keep_ids)] = ""
    return reason
