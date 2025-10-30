import streamlit as st

def chat_ui(state):
    st.header("質問‐応答チャット")
    for chat in state["history"]:
        if chat["role"] == "user":
            st.markdown(f"**あなた：** {chat['content']}")
        else:
            st.markdown(f"**AI：** {chat['content']}")
    user_input = st.chat_input("質問を入力してください")
    return user_input
