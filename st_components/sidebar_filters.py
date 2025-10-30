# sidebar_filters.py
import streamlit as st
import pandas as pd
from utils import load_jichitai, load_category

def sidebar_filters():
    """app.py のサイドバー構成を完全に踏襲した絞り込みUI関数。
       戻り値: 検索クエリ構築に必要な条件一式(dict)
    """

    # ====== データ読込 ======
    jichitai = load_jichitai("jichitai.xlsx")
    catmap = load_category("category.xlsx")

    # ====== Sidebar UI ======
    st.sidebar.subheader("🔍 キーワード・年度絞り込み")

    year_options = list(range(2010, 2031))
    selected_years = st.sidebar.multiselect(
        "年度（複数選択可）",
        options=year_options,
        default=[],
        help="fiscal_year_start/fiscal_year_endで絞り込み"
    )

    and_input = st.sidebar.text_input(
        "AND条件（スペース区切り）",
        placeholder="例: 環境 計画",
        help="全てのキーワードを含む文書を検索"
    )
    or_input = st.sidebar.text_input(
        "OR条件（スペース区切り）",
        placeholder="例: 温暖化 気候変動",
        help="いずれかのキーワードを含む文書を検索"
    )
    not_input = st.sidebar.text_input(
        "NOT条件（スペース区切り）",
        placeholder="例: 廃止 中止",
        help="これらのキーワードを含まない文書を検索"
    )

    search_title = st.sidebar.checkbox(
        "資料名も検索対象に含める",
        value=False,
        help="チェックを入れるとtitleフィールドも検索対象になります"
    )

    st.sidebar.markdown("---")

    # ========== 自治体絞り込み ==========
    st.sidebar.subheader("🔍 自治体・カテゴリ絞り込み")
    pref_opts = (
        jichitai[["affiliation_code", "pref_name"]]
        .drop_duplicates().assign(aff_num=lambda d: pd.to_numeric(d["affiliation_code"], errors="coerce"))
        .sort_values(["aff_num"])
    )
    sel_pref_names = st.sidebar.multiselect("都道府県", options=pref_opts["pref_name"].tolist())
    sel_aff_codes = pref_opts[pref_opts["pref_name"].isin(sel_pref_names)]["affiliation_code"].tolist()

    ctype_opts = sorted(jichitai["city_type"].dropna().unique().tolist())
    sel_city_types = st.sidebar.multiselect("自治体区分", options=ctype_opts)

    if sel_aff_codes:
        city_pool = jichitai[jichitai["affiliation_code"].isin(sel_aff_codes)]
    else:
        city_pool = jichitai.copy()
    if sel_city_types:
        city_pool = city_pool[city_pool["city_type"].isin(sel_city_types)]
    city_pool = city_pool.sort_values(["affiliation_code", "code"])
    sel_city_names = st.sidebar.multiselect("市区町村", options=city_pool["city_name"].tolist())
    sel_codes = city_pool[city_pool["city_name"].isin(sel_city_names)]["code"].tolist()

    cat_opts = catmap.sort_values("order")
    short_unique = cat_opts.drop_duplicates(subset=["short_name"], keep="first")
    sel_cat_short = st.sidebar.multiselect(
        "資料カテゴリ",
        options=short_unique["short_name"].tolist(),
        default=short_unique["short_name"].tolist()
    )
    sel_categories = cat_opts[cat_opts["short_name"].isin(sel_cat_short)]["category"].astype(int).tolist()

    # --- 戻り値整形 ---
    filters = {
        "years": selected_years,
        "and_input": and_input,
        "or_input": or_input,
        "not_input": not_input,
        "include_title": search_title,
        "pref_names": sel_pref_names,
        "city_types": sel_city_types,
        "city_names": sel_city_names,
        "codes": sel_codes,
        "categories": sel_categories,
    }

    return filters
