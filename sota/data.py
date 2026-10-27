"""Normalize IEEE records (API JSON or manual CSV exports) into one table."""
from __future__ import annotations

import hashlib
import re

import pandas as pd

# Same columns used by the original notebooks, plus identifiers.
COLUMNS = [
    "document_title", "publication_title", "publication_year", "abstract", "pdf_link",
    "document_identifier", "online_date", "authors", "author_affiliations", "ieee_terms",
    "author_keywords", "article_citation_count", "funding_information", "mesh_terms",
    "patent_citation_count",
]
EXTRA = ["id", "article_number", "doi", "html_url", "query"]

FILL = {
    "authors": "Unknown", "author_affiliations": "Unknown", "funding_information": "Unknown",
    "ieee_terms": "", "author_keywords": "", "mesh_terms": "",
    "article_citation_count": 0, "patent_citation_count": 0,
}


def _terms(index_terms: dict, key: str) -> str:
    return "; ".join(dict.fromkeys((index_terms or {}).get(key, {}).get("terms", [])))


def from_api(records: list[dict], query: str = "") -> pd.DataFrame:
    rows = []
    for a in records:
        authors = (a.get("authors") or {}).get("authors", [])
        it = a.get("index_terms") or {}
        rows.append({
            "document_title": a.get("title"),
            "publication_title": a.get("publication_title"),
            "publication_year": a.get("publication_year"),
            "abstract": a.get("abstract"),
            "pdf_link": a.get("pdf_url"),
            "document_identifier": f"{a.get('publisher', '')} {a.get('content_type', '')}".strip(),
            "online_date": a.get("publication_date"),
            "authors": "; ".join(x.get("full_name", "") for x in authors) or None,
            "author_affiliations": "; ".join(x["affiliation"] for x in authors if x.get("affiliation")) or None,
            "ieee_terms": _terms(it, "ieee_terms"),
            "author_keywords": _terms(it, "author_terms"),
            "article_citation_count": a.get("citing_paper_count"),
            "funding_information": None,  # not provided by the search API
            "mesh_terms": None,           # not provided by the search API
            "patent_citation_count": a.get("citing_patent_count"),
            "article_number": a.get("article_number"),
            "doi": a.get("doi"),
            "html_url": a.get("html_url"),
            "query": query,
        })
    return finalize(pd.DataFrame(rows))


def from_csv(files, query: str = "") -> pd.DataFrame:
    """IEEE Xplore 'Export > CSV' files or CSVs previously downloaded from the app."""
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = ["_".join(str(c).lower().split()) for c in df.columns]
    if "article_number" not in df and "pdf_link" in df:
        df["article_number"] = df["pdf_link"].astype(str).str.extract(r"arnumber=(\d+)")[0]
    if "query" not in df:
        df["query"] = query
    return finalize(df)


def _id(row) -> str:
    for k in ("article_number", "doi"):
        v = row.get(k)
        if pd.notna(v) and str(v).strip():
            return str(v).split(".")[0] if k == "article_number" else str(v)
    return hashlib.md5(str(row.get("document_title")).lower().encode()).hexdigest()[:12]


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep known columns, fill missing values, drop rows without abstract and duplicates."""
    for c in COLUMNS + EXTRA:
        if c not in df:
            df[c] = None
    df = df[COLUMNS + EXTRA].copy()
    df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 30)]
    df = df.fillna(FILL)
    for c in ("article_citation_count", "patent_citation_count"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    df["id"] = df.apply(_id, axis=1)

    # Same criteria as the notebooks: duplicated abstract -> most cited; duplicated title -> longest abstract.
    df = df.sort_values("article_citation_count", ascending=False)
    df = df[~_norm(df["abstract"]).duplicated()]
    df = df.assign(_len=df["abstract"].str.len()).sort_values("_len", ascending=False)
    df = df[~_norm(df["document_title"]).duplicated()].drop(columns="_len")
    df = df.drop_duplicates("id")
    return df.sort_index().reset_index(drop=True)


def merge(*dfs: pd.DataFrame) -> pd.DataFrame:
    return finalize(pd.concat([d for d in dfs if d is not None], ignore_index=True))


_TYPES = {"early access": "Early Access", "journal": "Journals", "conference": "Conferences",
          "magazine": "Magazines", "book": "Books", "standard": "Standards", "course": "Courses"}


def doc_type(df: pd.DataFrame) -> pd.Series:
    """'Journals', 'Conferences', ... from identifiers like 'IEEE Journals' or 'IET Conferences'."""
    s = df["document_identifier"].fillna("").str.lower()
    out = pd.Series("Other", index=df.index)
    for key, name in reversed(_TYPES.items()):  # first key wins ("early access" before "journal")
        out[s.str.contains(key, regex=False)] = name
    return out


def split_list(s: pd.Series, sep: str = ";") -> pd.Series:
    """Explode a '; '-separated column into one stripped value per row."""
    out = s.fillna("").str.split(sep).explode().str.strip()
    return out[(out != "") & (out != "Unknown")]


def country(affiliation: str) -> str:
    """Heuristic: last comma-separated chunk of an affiliation."""
    part = re.split(r",", affiliation)[-1].strip()
    return re.sub(r"\d+", "", part).strip()
