"""SOTA Explorer: state-of-the-art reviews from IEEE Xplore.

Run locally:  streamlit run streamlit_app.py
"""
import streamlit as st

from sota.ieee import IEEEClient, IEEEError

st.set_page_config(page_title="SOTA Explorer", page_icon="📚", layout="wide")


def login() -> None:
    st.title("📚 SOTA Explorer")
    st.write("Búsqueda en IEEE Xplore, limpieza asistida por clusters y visualización del estado del arte.")
    with st.form("login"):
        key = st.text_input("API key de IEEE Xplore", type="password",
                            help="Obtén una en https://developer.ieee.org. Solo se guarda en la memoria de "
                                 "esta sesión: no se escribe en disco ni en logs y se borra al cerrar sesión.")
        submitted = st.form_submit_button("Ingresar", type="primary")
    if submitted:
        try:
            IEEEClient(key.strip()).validate()
        except IEEEError as e:
            st.error(str(e))
            return
        st.session_state.api_key = key.strip()
        st.rerun()
    st.caption("🔒 La key viaja cifrada (HTTPS) de tu navegador a la app y de la app a IEEE. "
               "Cada usuario usa su propia key; la app no guarda ninguna.")
    st.divider()
    if st.button("Continuar sin API key (subir CSV exportados desde IEEE Xplore)"):
        st.session_state.api_key = ""
        st.rerun()


if "api_key" not in st.session_state:
    login()
    st.stop()

with st.sidebar:
    st.caption("Conectado a IEEE Xplore ✅" if st.session_state.api_key else "Modo sin API key")
    if st.button("Cerrar sesión", width="stretch"):
        st.session_state.clear()
        st.rerun()

st.navigation([
    st.Page("app_pages/search.py", title="1 · Buscar", icon="🔎"),
    st.Page("app_pages/clean.py", title="2 · Limpiar", icon="🧹"),
    st.Page("app_pages/explore.py", title="3 · Explorar", icon="📊"),
]).run()
