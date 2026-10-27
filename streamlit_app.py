"""SOTA Explorer: state-of-the-art reviews from IEEE Xplore.

Run locally:  streamlit run streamlit_app.py
"""
import streamlit as st

from sota.ieee import IEEEClient, IEEEError

st.set_page_config(page_title="SOTA Explorer", page_icon="📚", layout="wide")

GALLERY = st.Page("app_pages/gallery.py", title="SOTA publicados", icon="📚")
EXPLORE = st.Page("app_pages/explore.py", title="3 · Explorar", icon="📊")


def home() -> None:
    st.title("📚 SOTA Explorer")
    st.write("Búsqueda en IEEE Xplore, limpieza asistida por clusters y visualización del estado del arte.")
    left, right = st.columns(2, gap="large")

    with left.container(border=True):
        st.subheader("🔎 Revisar otros SOTA")
        st.write("Explora estados del arte ya procesados por otras personas. No necesitas API key.")
        if st.button("Ver SOTA publicados", type="primary", width="stretch"):
            st.switch_page(GALLERY)

    with right.container(border=True):
        st.subheader("🛠️ Crear un SOTA")
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
            else:
                st.session_state.api_key = key.strip()
                st.rerun()
        st.caption("🔒 La key viaja cifrada (HTTPS) de tu navegador a la app y de la app a IEEE. "
                   "Cada usuario usa su propia key; la app no guarda ninguna.")
        if st.button("Continuar sin API key (subir CSV exportados desde IEEE Xplore)"):
            st.session_state.api_key = ""
            st.rerun()


if "api_key" not in st.session_state:
    pages = [st.Page(home, title="Inicio", icon="🏠", default=True), GALLERY,
             st.Page("app_pages/explore.py", title="Explorar", icon="📊")]
else:
    pages = [
        st.Page("app_pages/search.py", title="1 · Buscar", icon="🔎", default=True),
        st.Page("app_pages/clean.py", title="2 · Limpiar", icon="🧹"),
        EXPLORE,
        GALLERY,
    ]
    with st.sidebar:
        st.caption("Conectado a IEEE Xplore ✅" if st.session_state.api_key else "Modo sin API key")
        if st.button("Cerrar sesión", width="stretch"):
            st.session_state.clear()
            st.rerun()

st.navigation(pages).run()
