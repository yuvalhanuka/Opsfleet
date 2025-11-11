import traceback
from datetime import datetime

from helper_functions import *
setup_logging()

import streamlit as st
from PIL import Image

from data_analysis_agent import DataAnalysisAgent, DataAnalysisAgentState

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Data Analysis Chat", page_icon="💬", layout="wide")
st.title("💬 Data Analysis Chat")
st.caption("Ask about the ecommerce data (`bigquery-public-data.thelook_ecommerce`). The agent may generate SQL and plots.")

with st.sidebar:
    st.header("⚙️ Options")
    show_debug = st.checkbox("Show debug/log", value=False)
    st.write("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ---------------- State ----------------
if "agent" not in st.session_state:
    st.session_state.agent = DataAnalysisAgent().get_data_analysis_agent()

if "chat" not in st.session_state:
    st.session_state.chat = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# ---------------- Helpers ----------------
def render_message(role: str, content: str, plot_path: str | None = None):
    """
    Render one chat bubble + optional plot image + download button.
    """
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content if content else "")
        if plot_path and os.path.exists(plot_path):
            st.image(Image.open(plot_path), caption=os.path.basename(plot_path), use_column_width=True)
            with open(plot_path, "rb") as f:
                st.download_button("Download plot", f, file_name=os.path.basename(plot_path), key=f"dl-{plot_path}-{os.path.getmtime(plot_path)}")
        elif plot_path:
            st.info(f"Plot path returned but file not found: `{plot_path}`")

def append_and_render(role: str, content: str, plot_path: str | None = None):
    st.session_state.chat.append({"role": role, "content": content, "plot_path": plot_path})
    render_message(role, content, plot_path)

# ---------------- Replay history ----------------
for m in st.session_state.chat:
    render_message(m["role"], m["content"], m.get("plot_path"))

# ---------------- Chat Input ----------------
prompt = st.chat_input("Type your question…")
if prompt:
    # 1) Show user message immediately
    append_and_render("user", prompt)

    # 2) Build agent state
    state: DataAnalysisAgentState = {
        "user_question": prompt,
        "messages": st.session_state.agent_messages,  # pass prior memory to agent
    }

    # 3) Invoke agent
    try:
        with st.spinner("Thinking…"):
            final_state = st.session_state.agent.invoke(state)
    except Exception as e:
        st.error("Sorry—something went wrong.")
        if show_debug:
            st.exception(e)
            st.code("".join(traceback.format_exception(e)), language="text")
        # Add an assistant error bubble
        append_and_render("assistant", "An error occurred. Please try again.")
    else:
        # 4) Persist agent memory + render assistant message and plot
        st.session_state.agent_messages = final_state.get("messages", st.session_state.agent_messages)
        answer = final_state.get("chat_response") or "_No answer returned._"
        plot_path = final_state.get("plot_file_path")
        append_and_render("assistant", answer, plot_path)

# ---------------- Debug Panel ----------------
if show_debug:
    st.divider()
    st.subheader("Debug")
    st.write("Agent memory/messages:")
    st.write(st.session_state.agent_messages)
