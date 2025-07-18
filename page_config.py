import streamlit as st

from config import (PAGE_TITLE, PAGE_ICON, APP_HEADING, APP_DESCRIPTION, APP_ICON)

def set_page_config():
    # Setup streamlit app
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="centered",
    )

    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title(f"{APP_ICON} {APP_HEADING}")
    st.write(APP_DESCRIPTION)
    st.markdown("<br>", unsafe_allow_html=True)
