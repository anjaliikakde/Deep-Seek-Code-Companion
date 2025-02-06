import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.markdown("""
<style>
    /* Main layout */
    .main { background-color: #121212; color: #E0E0E0; font-family: 'Calibri', sans-serif; }
    
    /* Sidebar styling */
    .sidebar .sidebar-content { background-color: #1E1E1E; }
    
    /* Input and select box */
    .stTextInput textarea, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        color: #E0E0E0 !important; background-color: #333333 !important; border-radius: 8px;
    }
    
    /* Chat bubbles */
    .stChatMessage { border-radius: 12px; padding: 12px; margin-bottom: 8px; }
    .stChatMessage[data-message-type="ai"] { background-color: #2A2A2A; color: #E0E0E0; }
    .stChatMessage[data-message-type="user"] { background-color: #333333; color: #E0E0E0; }
    
    /* Loading spinner */
    .stSpinner { color: #F39C12 !important; }
</style>
""", unsafe_allow_html=True)

# =====================[ HEADER & SIDEBAR CONFIGURATION ]=====================
st.title("🌐 DEEP-SEEK CODE COMPANION")
st.caption("🚀 AI-powered pair programmer for debugging, explanations, and coding assistance.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_model = st.selectbox("🔍 Choose AI Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
    
    st.divider()
    st.markdown("### 🔥 Features")
    st.markdown("""
    - 🐍 **Python Expert**
    - 🐞 **Bug Fixing**
    - 📝 **Code Documentation**
    - 💡 **Solution Design**
    """)

    st.divider()
    st.markdown("🔗 Built with [Ollama](https://ollama.ai/) & [LangChain](https://python.langchain.com/)")

# =====================[ FUNCTIONALITY: Load AI Model Efficiently ]=====================
@st.cache_resource
def load_llm_engine(model_name):
    """Load and cache the AI model."""
    return ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0.3)

llm_engine = load_llm_engine(selected_model)

# =====================[ SYSTEM PROMPT CONFIGURATION ]=====================
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a professional AI coding assistant. Provide concise, high-quality code solutions "
    "and suggest strategic debugging steps when necessary. Always respond in English."
)

# =====================[ SESSION STATE MANAGEMENT ]=====================
if "message_log" not in st.session_state:
    st.session_state["message_log"] = [{"role": "ai", "content": "Hello! I'm DeepSeek. How can I assist with your coding today? 💻"}]

# =====================[ FUNCTION: Construct Chat Prompt Sequence ]=====================
def build_prompt_chain():
    """Constructs a conversation sequence for AI context."""
    return ChatPromptTemplate.from_messages([
        system_prompt
    ] + [
        (HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user"
        else AIMessagePromptTemplate.from_template(msg["content"]))
        for msg in st.session_state.message_log
    ])

# =====================[ FUNCTION: Generate AI Response ]=====================
def generate_ai_response(prompt_chain):
    """Processes input through AI model and returns response."""
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# =====================[ CHAT INTERFACE: Display Previous Messages ]=====================
for message in st.session_state.message_log:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================[ USER INPUT HANDLING ]=====================
user_query = st.chat_input("💬 Ask your coding question...")

if user_query and user_query.strip():
    # Append user message
    st.session_state.message_log.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.spinner("🤖 Thinking..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Append AI response
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Refresh UI to display new messages
    st.rerun()
