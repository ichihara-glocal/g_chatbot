import streamlit as st
from utils import load_jichitai, load_category

def sidebar_filters():
    st.sidebar.header("絞り込み条件")
    keyword = st.sidebar.text_input("キーワード")
    years = load_category  # ← placeholder
    # 年度選択
    start_year = st.sidebar.number_input("開始年度", value= None, format="%d")
    end_year = st.sidebar.number_input("終了年度", value= None, format="%d")
    # 自治体
    jichitai_df = load_jichitai("jichitai.xlsx")
    jichitai = st.sidebar.selectbox("自治体", [""] + jichitai_df["名称"].tolist())
    # カテゴリ
    cat_df = load_category("category.xlsx")
    cat1 = st.sidebar.selectbox("カテゴリ①", [""] + cat_df["カテゴリ①"].unique().tolist())
    cat2 = None
    if cat1:
        filtered = cat_df[cat_df["カテゴリ①"] == cat1]
        cat2 = st.sidebar.selectbox("カテゴリ②", [""] + filtered["カテゴリ②"].tolist())
    if st.sidebar.button("検索開始"):
        return {
            "keyword": keyword,
            "start_year": start_year if start_year else None,
            "end_year": end_year if end_year else None,
            "jichitai": jichitai if jichitai else None,
            "category1": cat1 if cat1 else None,
            "category2": cat2 if cat2 else None
        }
    return None
