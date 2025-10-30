import streamlit as st
from st_components.sidebar_filters import sidebar_filters
from st_components.chat_ui import chat_ui
from es_client import get_es_client
from retrieval import search_documents, rank_by_embedding
from generation import generate_answer
from utils import load_jichitai, load_category

def main():
    st.set_page_config(page_title="G-Finder 対話型検索", layout="wide")
    st.sidebar.title("G-Finder 検索条件")
    filter_params = sidebar_filters()
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if filter_params:
        st.session_state["filters"] = filter_params
        st.session_state["history"] = []  # 絞り込み変更で履歴クリア

    user_input = chat_ui(st.session_state)

    if user_input:
        st.session_state["history"].append({"role": "user", "content": user_input})
        es_client = get_es_client()
        indices = st.secrets["es"]["indices"]

        docs = search_documents(
            es_client, indices,
            keyword=st.session_state["filters"]["keyword"],
            start_year=st.session_state["filters"]["start_year"],
            end_year=st.session_state["filters"]["end_year"],
            jichitai=st.session_state["filters"]["jichitai"],
            category1=st.session_state["filters"]["category1"],
            category2=st.session_state["filters"]["category2"],
            top_n=10000
        )
        ranked_docs = rank_by_embedding(user_input, docs, top_k=100)
        answer = generate_answer(user_input, ranked_docs, st.secrets["gemini"]["api_key"])
        st.session_state["history"].append({"role": "ai", "content": answer})

    # 表示
    for chat in st.session_state["history"]:
        if chat["role"] == "user":
            st.markdown(f"**あなた：** {chat['content']}")
        else:
            st.markdown(f"**AI：** {chat['content']}")

if __name__ == "__main__":
    main()
