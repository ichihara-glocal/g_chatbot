import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

@st.cache_data
def load_jichitai(filepath: str):
    df = pd.read_excel(filepath, engine="openpyxl")
    return df

@st.cache_data
def load_category(filepath: str):
    df = pd.read_excel(filepath, engine="openpyxl")
    return df

def embed_text(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings
