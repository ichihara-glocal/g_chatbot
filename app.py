# app.py
import streamlit as st
from st_components.sidebar_filters import sidebar_filters
from st_components.chat_ui import chat_ui
from es_client import get_es_client
from retrieval import build_search_query, search_documents, rank_by_embedding
from generation import generate_answer
from utils import load_jichitai, load_category
import datetime

# ==========================================================
# ================ Streamlit 基本設定 =======================
# ==========================================================

st.set_page_config(page_title="G-Finder 対話型AI検索", layout="wide")
st.title("質問・応答チャット")

# ==========================================================
# ================ 初期状態管理 =============================
# ==========================================================

if "history" not in st.session_state:
    st.session_state["history"] = []

if "filters" not in st.session_state:
    st.session_state["filters"] = {}

# ==========================================================
# ================ Sidebar：絞り込み条件 ====================
# ==========================================================

filters = sidebar_filters()
if filters:
    # フィルタ変更時は履歴をリセット
    st.session_state["filters"] = filters
    st.session_state["history"] = []

# ==========================================================
# ================ チャットUI部 =============================
# ==========================================================

user_input = chat_ui(st.session_state)

# ==========================================================
# ================ 質問送信時の処理 =========================
# ==========================================================

if user_input:
    # 履歴追加
    st.session_state["history"].append({"role": "user", "content": user_input})

    # --- ESクライアント初期化 ---
    es_client = get_es_client()
    indices = st.secrets["es"]["indices"]
    gemini_api_key = st.secrets["gemini"]["api_key"]

    # --- 検索クエリ構築 ---
    filters = st.session_state.get("filters", {})
    query = build_search_query(filters)

    # --- Elasticsearch検索 ---
    st.info("🔍 G-Finderから関連ドキュメントを検索中...")
    docs = search_documents(es_client, indices, query, limit=10000)

    if not docs:
        st.warning("該当するドキュメントが見つかりませんでした。条件を見直してください。")
    else:
        # --- Gemini Embedding による類似度ランキング ---
        st.info("✨ Geminiによる関連度分析中...")
        ranked_docs = rank_by_embedding(
            question=user_input,
            docs=docs,
            api_key=gemini_api_key,
            embedding_model="models/text-embedding-004",  # 安定版
            top_k=100
        )

        if not ranked_docs:
            st.warning("関連ドキュメントの類似度分析に失敗しました。")
        else:
            # --- Gemini 2.0 Flash による回答生成 ---
            st.info("🤖 Gemini 2.0 Flash で回答を生成中...")
            try:
                answer = generate_answer(
                    question=user_input,
                    docs=ranked_docs,
                    api_key=gemini_api_key,
                    model="gemini-2.0-flash",
                    max_output_tokens=800
                )
                st.session_state["history"].append({"role": "ai", "content": answer})
            except Exception as e:
                st.error(f"Gemini回答生成中にエラーが発生しました: {e}")

# ==========================================================
# ================ チャット履歴表示 =========================
# ==========================================================

for chat in st.session_state["history"]:
    if chat["role"] == "user":
        st.markdown(f"**あなた：** {chat['content']}**")
    else:
        st.markdown(f"**AI：** {chat['content']}**")

# ==========================================================
# ================ 注意書き ================================
# ==========================================================

st.markdown("""
---
**注意事項**
- 検索対象は G-Finder に格納された公開文書に基づきます。
- AI回答は参考情報であり、最終判断は必ず原文をご確認ください。
- 一度に扱う文書数が多い場合、処理に時間がかかることがあります。
- 通信環境や外部APIの制限により、まれに処理が中断することがあります。
""")
