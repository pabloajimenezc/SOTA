import base64

import pandas as pd
import pytest

from sota import data, store
from tests.test_core import FakeResp, article


class FakeGitHub:
    """In-memory GitHub contents API + raw.githubusercontent.com."""

    def __init__(self):
        self.files = {}

    def request(self, method, url, timeout, params=None, json=None, headers=None):
        path = url.split("/contents/")[-1] if "/contents/" in url else url.split("/sotas/", 1)[-1]
        if method == "PUT":
            assert headers["Authorization"] == "Bearer TOKEN"
            assert (json.get("sha") is not None) == (path in self.files)  # updates must send the sha
            self.files[path] = base64.b64decode(json["content"])
            return FakeResp({}, 201)
        if path not in self.files:
            return FakeResp({}, 404)
        if headers and headers["Accept"].endswith("+json"):
            return FakeResp({"sha": "abc"})
        r = FakeResp({})
        r.content = self.files[path]
        return r


def sample():
    df = data.from_api([article(i, year=2010 + i) for i in range(1, 6)], "q")
    return df.assign(x=range(5), y=range(5))


def test_publish_load_and_owner_protection():
    gh = FakeGitHub()
    s = store.GitHubStore("me/repo", token="TOKEN", session=gh)
    df = sample()
    meta = store.make_meta(df, "Electric Ships ñ", "desc", "Ana", store.owner_id("KEY1"), 6, {"synonyms": {}})
    assert meta["slug"] == "electric-ships-n" and meta["years"] == [2011, 2015] and meta["queries"] == ["q"]
    s.save(df, meta)
    s.save(df, meta)  # same owner can republish
    assert [m["slug"] for m in s.index()] == ["electric-ships-n"]
    back = s.load("electric-ships-n")
    pd.testing.assert_frame_equal(back, df[[c for c in store.SAVE_COLUMNS if c in df]], check_dtype=False)

    other = meta | {"owner": store.owner_id("KEY2")}
    with pytest.raises(store.StoreError, match="otra persona"):
        s.save(df, other)

    ro = store.GitHubStore("me/repo", session=gh)  # no token: reads only
    assert not ro.writable and ro.index()[0]["name"] == "Electric Ships ñ"
    with pytest.raises(store.StoreError, match="token"):
        ro.save(df, meta)
    assert "TOKEN" not in repr(s)
