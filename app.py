import streamlit as st
from groq import Groq

# --- 페이지 설정 ---
st.set_page_config(page_title="Qwen3 Chatbot", page_icon="🤖")
st.title("🤖 Qwen3-32b Chat (powered by Groq)")

# --- Groq 클라이언트 초기화 ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("API 키가 설정되지 않았습니다. Streamlit 대시보드에서 Secrets를 확인해주세요.")
    st.stop()

# 사용할 모델 설정 
MODEL_ID = "qwen/qwen3-32b" 

# --- 세션 상태(Session State) 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 기존 대화 내역 출력 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            response = st.write_stream(
                (chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content)
            )
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
