from elasticsearch import Elasticsearch
import streamlit as st

def get_es_client():
    es_host = st.secrets["es"]["host"]
    es_user = st.secrets["es"]["user"]
    es_password = st.secrets["es"]["password"]
    client = Elasticsearch(
        hosts=[es_host],
        http_auth=(es_user, es_password),
        verify_certs=True
    )
    return client
