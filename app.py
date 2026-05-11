import streamlit as st
from groq import Groq
import re

# --- 1. 페이지 설정 및 커스텀 디자인 ---
st.set_page_config(page_title="Qwen Chatbot", page_icon="✨", layout="centered")

# CSS를 사용해 타이틀과 채팅 말풍선 디자인을 깔끔하게 다듬습니다.
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    /* 사용자 채팅 말풍선 배경색 */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 15px;
    }
    /* AI 채팅 말풍선 배경색 및 테두리 */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 15px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>✨ Qwen3-32b Smart Chat</h1>", unsafe_allow_html=True)

# --- 2. 시스템 프롬프트 (AI 규칙 설정) ---
SYSTEM_PROMPT = """
당신은 논리적이고 유용한 AI 어시스턴트입니다. 다음 규칙을 엄격히 지키세요:
1. 이모티콘을 절대 사용하지 마십시오.
2. 불필요한 마크다운(과도한 볼드체 등) 사용을 자제하고 깔끔한 평문 위주로 작성하십시오.
3. 코드 작성이나 특수한 케이스(영어 단어 설명 등)를 제외하고는 무조건 '한국어'로만 답변하십시오.
"""

# --- Groq 클라이언트 초기화 ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("API 키가 설정되지 않았습니다. Streamlit 대시보드에서 Secrets를 확인해주세요.")
    st.stop()

MODEL_ID = "qwen/qwen3-32b"

# --- 3. <think> 태그 필터링 제너레이터 ---
# AI가 생각하는 과정(스트리밍 데이터)을 가로채서 화면에 안 보이게 처리합니다.
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
                    yield buffer[:start_idx] # <think> 이전 텍스트만 화면에 출력
                    buffer = buffer[start_idx + len("<think>"):]
                    in_think = True
                else:
                    last_lt = buffer.rfind("<")
                    if last_lt != -1 and len(buffer) - last_lt < 8:
                        # <think> 태그가 잘려서 들어올 경우를 대비해 대기
                        yield buffer[:last_lt]
                        buffer = buffer[last_lt:]
                        break 
                    else:
                        yield buffer
                        buffer = ""
            else: # in_think == True
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    buffer = buffer[end_idx + len("</think>"):]
                    in_think = False
                else:
                    if len(buffer) > 8:
                        buffer = buffer[-8:] # </think>를 찾기 위해 끝부분만 남김
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
            # 규칙(System Prompt)을 메시지 맨 앞에 몰래 끼워넣습니다.
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            api_messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])

            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=api_messages,
                stream=True,
            )
            
            # 필터링 함수를 거쳐 화면에 출력합니다.
            filtered_stream = parse_stream(stream)
            response = st.write_stream(filtered_stream)
            
            # 최종 정제된 텍스트만 세션에 저장 (토큰 절약 효과)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
