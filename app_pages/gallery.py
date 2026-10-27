import json

import streamlit as st

from app_pages.shared import rules, set_raw, sota_index, sota_load
from sota.cleaning import Rules
from sota.store import StoreError

st.header("📚 SOTA publicados")
st.caption("Estados del arte ya procesados (búsqueda + limpieza). Ábrelos para explorarlos; no necesitas API key.")


def open_sota(m: dict) -> None:
    if st.session_state.get("raw") is not None and "sota_meta" not in st.session_state:
        st.session_state.my_work = {"raw": st.session_state.raw, "rules": rules()}
    try:
        df = sota_load(m["slug"], f"{m.get('created')}:{m.get('bytes')}")
    except StoreError as e:
        st.error(str(e))
        return
    set_raw(df)
    st.session_state.rules = Rules.from_json(json.dumps(m.get("rules") or {}))
    st.session_state.sota_meta = m
    st.session_state.explore_view = "Resumen"
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
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"**{m['name']}**  \n{m.get('description', '')}")
        years = "–".join(map(str, m["years"])) if m.get("years") else "?"
        c1.caption(f"{m['docs']:,} artículos · {years} · {m.get('author') or 'anónimo'} · "
                   f"publicado {m.get('created', '?')} · {m.get('bytes', 0) / 1e6:.1f} MB")
        if c2.button("Abrir", key=f"open_{m['slug']}", type="primary", width="stretch"):
            open_sota(m)
