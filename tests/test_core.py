import json

import numpy as np
import pandas as pd
import pytest
import requests

from sota import cleaning, data, ieee, text


def article(i, year=2020, cites=0, abstract=None, ctype="Journals"):
    return {
        "title": f"Title {i}", "publication_title": "IEEE Trans X", "publication_year": year,
        "abstract": abstract or f"Electric ship propulsion with battery storage number {i} and converters.",
        "pdf_url": f"https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber={i}", "content_type": ctype,
        "publisher": "IEEE", "article_number": str(i), "citing_paper_count": cites, "rank": i,
        "authors": {"authors": [{"full_name": "Ana Pérez", "affiliation": "Univ. X, Santiago, Chile"}]},
        "index_terms": {"ieee_terms": {"terms": ["Marine vehicles", "Batteries"]},
                        "author_terms": {"terms": ["ship", "ship"]}},
    }


class FakeResp:
    def __init__(self, payload, status=200):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p


class FakeSession:
    """Serves a fixed corpus honoring start_record/max_records/start_year/end_year/content_type."""

    def __init__(self, corpus):
        self.corpus, self.calls = corpus, []

    def get(self, url, params, timeout):
        self.calls.append(params)
        docs = [a for a in self.corpus
                if params.get("start_year", 0) <= a["publication_year"] <= params.get("end_year", 9999)
                and params.get("content_type", a["content_type"]) == a["content_type"]]
        s, m = params.get("start_record", 1) - 1, params["max_records"]
        return FakeResp({"total_records": len(docs), "articles": docs[s:s + m]})


def client(corpus):
    return ieee.IEEEClient("SECRET", session=FakeSession(corpus), pause=0)


def test_paginates_by_relevance():
    c = client([article(i) for i in range(1, 451)])
    recs = c.search(ieee.Query("x"), 450, counts={None: 450})
    assert len(recs) == 450 and c.calls == 3


def test_newest_splits_years():
    corpus = [article(i, year=2000 + i % 25) for i in range(1, 501)]
    recs = client(corpus).search(ieee.Query("x", start_year=2000, end_year=2024), 50, "Más recientes",
                                 counts={None: 500})
    years = [a["publication_year"] for a in recs]
    assert len(recs) == 50 and years == sorted(years, reverse=True) and min(years) >= 2022


def test_cited_sorts_locally_and_types_are_proportional():
    corpus = [article(i, cites=i, ctype="Journals" if i % 4 else "Conferences") for i in range(1, 201)]
    c = client(corpus)
    counts = c.counts(ieee.Query("x"), ["Journals", "Conferences"])
    assert counts == {"Journals": 150, "Conferences": 50}
    recs = c.search(ieee.Query("x"), 40, "Más citados", counts=counts)
    assert len(recs) == 40 and recs[0]["citing_paper_count"] == 200
    assert sum(a["content_type"] == "Conferences" for a in recs) == 10


def test_errors_never_leak_key():
    class Boom:
        def get(self, url, params, timeout):
            raise requests.ConnectionError(f"failed {url}?apikey={params['apikey']}")

    with pytest.raises(ieee.IEEEError) as e:
        ieee.IEEEClient("SECRET", session=Boom(), pause=0).validate()
    assert "SECRET" not in str(e.value) and e.value.__cause__ is None
    assert "SECRET" not in repr(ieee.IEEEClient("SECRET"))

    class Forbidden:
        def get(self, url, params, timeout):
            return FakeResp({}, 403)

    with pytest.raises(ieee.IEEEError, match="inválida"):
        ieee.IEEEClient("SECRET", session=Forbidden(), pause=0).validate()


def test_from_api_maps_columns_and_dedupes():
    recs = [article(1), article(2), article(3, abstract=article(1)["abstract"], cites=5), article(4, abstract="short")]
    df = data.from_api(recs, "q")
    assert list(df.columns) == data.COLUMNS + data.EXTRA
    assert len(df) == 2  # 3 duplicates 1's abstract (keeps the most cited), 4 has no real abstract
    assert set(df["id"]) == {"2", "3"}
    row = df.iloc[0]
    assert row["ieee_terms"] == "Marine vehicles; Batteries" and row["author_keywords"] == "ship"
    assert row["document_identifier"] == "IEEE Journals" and row["funding_information"] == "Unknown"
    assert data.doc_type(df).iloc[0] == "Journals"


def test_from_csv_manual_export(tmp_path):
    p = tmp_path / "export.csv"
    pd.DataFrame({"Document Title": ["A ship"], "Abstract": ["x" * 50], "Publication Year": [2021],
                  "PDF Link": ["https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=123"],
                  "Document Identifier": ["IEEE Conferences"]}).to_csv(p, index=False)
    df = data.from_csv([p])
    assert df.loc[0, "id"] == "123" and data.doc_type(df).iloc[0] == "Conferences"


def test_rules_syntax_and_overrides():
    df = pd.DataFrame({"id": list("abcde"), "publication_year": [2020, 2020, 2020, 2010, 2020],
                       "document_title": ["Blood vessel imaging", "Ship propulsion", "Plasmatic flow",
                                          "Electric ferry", "Cardiovascular ship"],
                       "abstract": ["", "electric motor drive", "", "battery", "electric"],
                       "ieee_terms": "", "author_keywords": "", "mesh_terms": ""})
    r = cleaning.Rules(exclude_terms=["blood vessel", "plasma*", "cardio*"], require_terms=["electric*", "batter*"],
                       min_year=2015, keep_ids=["e"])
    reason = cleaning.apply(df, r)
    assert reason.tolist() == ["término: blood vessel", "", "término: plasmatic", "año < 2015", ""]
    r2 = cleaning.Rules.from_json(r.to_json())
    assert r2 == r


def test_text_pipeline():
    topics = ["ship battery propulsion electric motor", "heart blood vessel artery medical imaging"]
    df = pd.DataFrame({"document_title": [f"{topics[i % 2]} study {i}" for i in range(40)],
                       "abstract": [topics[i % 2] * 3 for i in range(40)], "ieee_terms": "", "author_keywords": "",
                       "mesh_terms": "", "publication_year": [2000 + i % 20 for i in range(40)]})
    v = text.vectorize(df, {"vessel": "ship"})
    labels = text.cluster(v.Z, 2)
    assert len(set(labels[::2])) == 1 and labels[0] != labels[1]
    summ = text.cluster_summary(v, labels, df)
    assert set(summ["cluster"]) == {0, 1}
    sim = text.similar(v, ["electric ship"])
    assert np.argmax(sim[0]) % 2 == 0
    assert text.clean("Vessels and ships", {"vessel": "ship"}) == "ship ship"
