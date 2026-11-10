import json

import streamlit as st

from app_pages.shared import rules, set_raw, sota_index, sota_load, sota_version
from sota.cleaning import Rules
from sota.store import StoreError

st.title("SOTA publicados")
st.caption("Estados del arte ya procesados y revisados. Abre uno para explorar sus temas, mapa, autores y tendencias.")


def open_sota(m: dict) -> None:
    if st.session_state.get("raw") is not None and "sota_meta" not in st.session_state:
        st.session_state.my_work = {"raw": st.session_state.raw, "rules": rules()}
    try:
        df = sota_load(m["slug"], sota_version(m))
    except StoreError as e:
        st.error(str(e))
        return
    set_raw(df)
    st.session_state.rules = Rules.from_json(json.dumps(m.get("rules") or {}))
    st.session_state.sota_meta = m
    st.session_state.k_topics = m.get("k", 10)
    st.switch_page("app_pages/explore.py")


if mine := st.session_state.get("my_work"):
    if st.button("↩️ Volver a mis datos"):
        set_raw(mine["raw"])
        st.session_state.rules = mine["rules"]
        del st.session_state.my_work
        st.switch_page("app_pages/explore.py")

try:
    idx = sota_index()
except StoreError as e:
    st.error(str(e))
    st.stop()

q = st.text_input("Buscar", placeholder="Nombre, descripción o autor").lower()
shown = [m for m in sorted(idx, key=lambda m: m.get("created", ""), reverse=True)
         if q in f"{m['name']} {m.get('description', '')} {m.get('author', '')}".lower()]
if not shown:
    st.info("No hay SOTA publicados." if not idx else "Sin coincidencias.")

for m in shown:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"#### {m['name']}")
        c1.write(m.get("description", ""))
        years = "–".join(map(str, m["years"])) if m.get("years") else "?"
        badges = [f":blue-badge[{m['docs']:,} artículos]", f":gray-badge[{years}]"]
        if m.get("chile"):
            badges.append(f":red-badge[★ {m['chile']} con autores en Chile]")
        c1.markdown(" ".join(badges) + f"  \n:gray[Por {m.get('author') or 'anónimo'} · {m.get('created', '?')}]")
        if c2.button("Abrir", key=f"open_{m['slug']}", type="primary", width="stretch"):
            open_sota(m)
