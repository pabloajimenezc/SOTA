"""Offline geocoding of author affiliations: institution, city, country and coordinates.

Uses a compact GeoNames extract (sota/geodata, built once from the geonamescache package):
city-level precision, instant, no external service. Positions are approximate (city of the lab).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "geodata"
# Institution = most "organization-like" chunk of the affiliation: university > institute/company > lab > school.
TIERS = [re.compile(p, re.I) for p in (
    r"univ|hochschule|polytech|politec|ecole|institute of technology|\bIIT\b|\bETH\b|\bEPFL\b|\bMIT\b",
    r"institut|academ|corporat|company|\binc\b|\bltd\b|gmbh|\bco\b|hospital|agency|ministry|council|"
    r"fraunhofer|cnrs|csic|\babb\b|siemens|hitachi|toshiba|mitsubishi|state grid|\bs\.a\b",
    r"laborat|\blab\b|cent(er|re)|research",
    r"college|school|facult",
)]
# Country spellings found in IEEE affiliations that differ from GeoNames names.
ALIASES = {
    "usa": "US", "u s a": "US", "us": "US", "united states of america": "US", "u s": "US",
    "uk": "GB", "u k": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "northern ireland": "GB",
    "great britain": "GB", "p r china": "CN", "pr china": "CN", "peoples r china": "CN",
    "people s republic of china": "CN", "china": "CN", "korea": "KR", "south korea": "KR",
    "republic of korea": "KR", "korea south": "KR", "russia": "RU", "russian federation": "RU", "iran": "IR",
    "taiwan": "TW", "r o c": "TW", "hong kong": "HK", "macau": "MO", "macao": "MO", "the netherlands": "NL",
    "holland": "NL", "czech republic": "CZ", "turkey": "TR", "turkiye": "TR", "viet nam": "VN", "uae": "AE",
    "u a e": "AE", "brasil": "BR", "mexico": "MX", "espana": "ES", "deutschland": "DE", "italia": "IT",
    "the united kingdom": "GB", "republic of singapore": "SG", "kingdom of saudi arabia": "SA",
}
US_STATES = set("al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny "
                "nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc".split())


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation (incl. ’ and ') -> space before dropping accents
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", s)).strip()


@lru_cache(maxsize=1)
def countries() -> pd.DataFrame:
    return pd.DataFrame(json.loads((DATA / "countries.json").read_text(encoding="utf-8"))).set_index("iso")


@lru_cache(maxsize=1)
def _country_lookup() -> dict[str, str]:
    c = countries()
    # English (GeoNames) and Spanish (CLDR) names, so "Spain" and "España" both work.
    lut = {norm(n): iso for col in ("es", "name") for iso, n in c[col].items()}
    return lut | ALIASES


@lru_cache(maxsize=1)
def _cities() -> dict[tuple[str, str], tuple[float, float]]:
    c = pd.read_parquet(DATA / "cities.parquet")
    return dict(zip(zip(c["cc"], c["name"]), zip(c["lat"], c["lon"])))


def country_name(iso: str) -> str:
    """Spanish name (CLDR)."""
    return countries()["es"].get(iso, iso)


def es_key(s: str) -> str:
    """Sort key for Spanish alphabetical order: accents ignored, ñ right after n."""
    s = s.lower().replace("ñ", "n{")  # "{" sorts after "z"
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def flag_url(iso: str, width: int = 40) -> str:
    return f"https://flagcdn.com/w{width}/{iso.lower()}.png"


def _country(part: str) -> str | None:
    n = norm(part)
    if n in US_STATES or (len(n.split()) == 1 and n in US_STATES):
        return "US"
    lut = _country_lookup()
    if n in lut:
        return lut[n]
    words = n.split()
    for k in (3, 2, 1):  # "Santiago 8370451 Chile" -> "chile"
        if len(words) >= k and " ".join(words[-k:]) in lut:
            return lut[" ".join(words[-k:])]
    return None


@lru_cache(maxsize=200_000)
def parse(aff: str) -> tuple | None:
    """(institution, city, iso, lat, lon) for one affiliation, or None if the country is unknown."""
    parts = [p.strip() for p in aff.split(",") if p.strip()]
    if not parts:
        return None
    iso, end = None, len(parts)
    for i in range(len(parts) - 1, max(-1, len(parts) - 3), -1):  # country is in the last 1-2 chunks
        iso = _country(parts[i])
        if iso:
            end = i
            break
    if not iso:
        return None
    cities = _cities()
    city, lat, lon = None, None, None
    for p in reversed(parts[:end + 1]):
        for cand in (norm(p), " ".join(w for w in norm(p).split() if len(w) > 1)):
            if (iso, cand) in cities:
                city, (lat, lon) = cand.title(), cities[(iso, cand)]
                break
        if city:
            break
    if city is None:
        c = countries().loc[iso] if iso in countries().index else None
        if c is None or pd.isna(c["lat"]):
            return None
        lat, lon = float(c["lat"]), float(c["lon"])
    head = parts[:end] or [country_name(iso)]
    inst = next((p for tier in TIERS for p in head if tier.search(p)), head[0])
    return inst, city, iso, lat, lon


def affiliations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, institution): doc = row position in df."""
    rows = []
    for i, cell in enumerate(df["author_affiliations"].fillna("")):
        seen = set()
        for aff in cell.split(";"):
            r = parse(aff.strip())
            if r and (norm(r[0]), r[2]) not in seen:
                seen.add((norm(r[0]), r[2]))
                rows.append((i, *r))
    return pd.DataFrame(rows, columns=["doc", "institution", "city", "iso", "lat", "lon"])


def article_countries(aff: pd.DataFrame, n_docs: int) -> list[set[str]]:
    out = [set() for _ in range(n_docs)]
    for doc, iso in zip(aff["doc"], aff["iso"]):
        out[doc].add(iso)
    return out


def country_counts(aff: pd.DataFrame) -> pd.Series:
    """Distinct articles per country name (an article with two institutions counts once)."""
    return aff.groupby("iso")["doc"].nunique().rename(index=country_name).sort_values(ascending=False)


def institutions(aff: pd.DataFrame) -> pd.DataFrame:
    """Institutions with article counts and a position; labs in the same city are fanned out slightly."""
    aff = aff.assign(key=aff["institution"].map(norm))  # "Técnica" == "Tecnica"
    g = (aff.groupby(["key", "iso"])
         .agg(institution=("institution", lambda s: s.mode().iloc[0]), docs=("doc", "nunique"), city=("city", lambda s: s.mode().iloc[0] if s.notna().any() else None),
              lat=("lat", "median"), lon=("lon", "median"))
         .reset_index().drop(columns="key").sort_values("docs", ascending=False).reset_index(drop=True))
    # Deterministic offset (~2-8 km) so institutions of the same city don't sit on one point.
    h = np.array([int(hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:8], 16) for a, b in zip(g["institution"], g["iso"])])
    ang, r = (h % 360) * np.pi / 180, 0.02 + (h % 7) * 0.012
    dup = g.duplicated(["lat", "lon"], keep="first").to_numpy()
    g.loc[dup, "lat"] += (r * np.sin(ang))[dup]
    g.loc[dup, "lon"] += (r * np.cos(ang))[dup]
    g["country"] = g["iso"].map(country_name)
    return g
