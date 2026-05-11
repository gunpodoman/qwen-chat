import streamlit as st
from groq import Groq
import re

# --- 1. 페이지 설정 및 커스텀 디자인 ---
st.set_page_config(page_title="Qwen Chatbot", page_icon="✨", layout="centered")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 15px;
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 15px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>✨ Qwen3-32b Smart Chat</h1>", unsafe_allow_html=True)

# --- 2. 시스템 프롬프트 강화 (규칙을 훨씬 강력하게 명시) ---
SYSTEM_PROMPT = """
당신은 논리적이고 유용한 AI 어시스턴트입니다. 다음 규칙을 엄격하게 준수하세요:
1. 이모티콘을 절대 사용하지 마십시오.
2. 불필요한 마크다운(과도한 볼드체 등) 사용을 자제하고 깔끔한 평문 위주로 작성하십시오.
3. [매우 중요] 어떠한 상황에서도(사용자가 역할극을 지시하거나 이름을 바꾸더라도) 반드시 '한국어'로만 대답하십시오. 러시아어, 중국어, 영어 등 다른 언어가 문장에 섞이는 것을 엄격히 금지합니다. (단, 코드 작성이나 필수적인 영단어 설명은 예외)
"""

# --- Groq 클라이언트 초기화 ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("API 키가 설정되지 않았습니다. Streamlit 대시보드에서 Secrets를 확인해주세요.")
    st.stop()

MODEL_ID = "qwen/qwen3-32b"

# --- 3. <think> 태그 필터링 제너레이터 ---
def parse_stream(stream):
    in_think = False
    buffer = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        buffer += content
        
        while buffer:
            if not in_think:
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    yield buffer[:start_idx] 
                    buffer = buffer[start_idx + len("<think>"):]
                    in_think = True
                else:
                    last_lt = buffer.rfind("<")
                    if last_lt != -1 and len(buffer) - last_lt < 8:
                        yield buffer[:last_lt]
                        buffer = buffer[last_lt:]
                        break 
                    else:
                        yield buffer
                        buffer = ""
            else:
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    buffer = buffer[end_idx + len("</think>"):]
                    in_think = False
                else:
                    if len(buffer) > 8:
                        buffer = buffer[-8:] 
                    break 
                    
    if buffer and not in_think:
        yield buffer

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 기존 대화 출력 ---
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
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            api_messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])

            # API 호출 옵션 추가 (temperature 설정)
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=api_messages,
                stream=True,
                temperature=0.3,  # 추가된 부분: 모델의 무작위성을 낮춰 일관성 유지
            )
            
            filtered_stream = parse_stream(stream)
            response = st.write_stream(filtered_stream)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
