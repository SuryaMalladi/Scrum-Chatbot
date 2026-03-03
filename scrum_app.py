import os
import bs4
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Scrum Master Chat Bot",
    page_icon="🏃",
    layout="centered"
)

# ============================================================
# CUSTOM CSS - Blue and White color palette
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #f0f7ff;
}

.header-banner {
    background: linear-gradient(135deg, #1565c0 0%, #1e88e5 50%, #42a5f5 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(30, 136, 229, 0.3);
    text-align: center;
}
.header-banner h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}
.header-banner p {
    color: rgba(255,255,255,0.9);
    margin: 0.4rem 0 0 0;
    font-size: 1rem;
}

.user-message {
    background: linear-gradient(135deg, #1e88e5, #1565c0);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 1rem;
    line-height: 1.6;
}

.bot-message {
    background: white;
    color: #1a2940;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 1rem;
    line-height: 1.6;
    border: 1px solid #bbdefb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.msg-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
}
.user-label { color: #1565c0; text-align: right; }
.bot-label  { color: #42a5f5; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
}

.stTextInput > div > div > input {
    border: 2px solid #90caf9;
    border-radius: 12px;
    font-size: 1rem;
    background-color: white;
    color: #1a2940;
}
.stTextInput > div > div > input:focus {
    border-color: #1e88e5;
}

.stButton > button {
    background: linear-gradient(135deg, #1e88e5, #1565c0);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    padding: 0.6rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Get free key at console.groq.com",
        placeholder="gsk_..."
    )

    st.markdown("---")
    st.markdown("### 💡 Try These Questions")
    questions = [
        "What is a Sprint?",
        "Who are the Scrum Team members?",
        "How long is a Sprint Review?",
        "What is the Definition of Done?",
        "What happens in Sprint Planning?"
    ]
    for q in questions:
        if st.button(q, key=q):
            st.session_state["quick_q"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#1565c0;'>
    📖 Knowledge Base: Scrum Guide 2020<br>
    🤖 LLM: Groq Llama 3.1 (Free)<br>
    🧠 Memory: Last 3 conversations
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class='header-banner'>
    <h1>🏃 Scrum Master Chat Bot</h1>
    <p>Powered by Scrum Guide 2020 • Ask me anything about Scrum!</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# BUILD RAG SYSTEM (runs once, cached)
# ============================================================
@st.cache_resource
def build_rag(api_key):
    # Load Scrum Guide from web
    loader = WebBaseLoader(
        web_paths=("https://scrumguides.org/scrum-guide.html",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="content"))
    )
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    # Create embeddings (free, local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Setup Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.1,
        max_tokens=800
    )

    # Prompt template
    template = """You are an expert Scrum Master assistant. Answer based ONLY on the Scrum Guide 2020.
If the answer is not in the context, say: "This topic isn't covered in the Scrum Guide 2020."
Be clear, concise and use bullet points where helpful.

Conversation History (last 3 exchanges):
{chat_history}

Relevant sections from Scrum Guide 2020:
{context}

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def format_history(messages):
        if not messages:
            return "No previous conversation."
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        return "\n".join(formatted)

    # Build RAG chain
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_history(x["chat_history"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ============================================================
# INITIALIZE RAG
# ============================================================
if groq_api_key:
    if not st.session_state.rag_ready:
        with st.spinner("📚 Loading Scrum Guide 2020... Please wait (~30 seconds)"):
            try:
                st.session_state.rag_chain = build_rag(groq_api_key)
                st.session_state.rag_ready = True
                st.success("✅ Ready! Ask me anything about Scrum.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👈 Enter your FREE Groq API key in the sidebar to get started. Get one at console.groq.com")

# ============================================================
# DISPLAY CHAT MESSAGES
# ============================================================
st.markdown("### 💬 Chat")
st.markdown("---")

if not st.session_state.messages:
    st.markdown("""
    <div class='msg-label bot-label'>🤖 Scrum Bot</div>
    <div class='bot-message'>
        Hello! I'm your Scrum Master assistant, trained on the <b>Scrum Guide 2020</b>.<br><br>
        Ask me anything about Sprints, Scrum roles, ceremonies, artifacts, and more!
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='msg-label user-label'>👤 You</div>
        <div class='user-message'>{msg['content']}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='msg-label bot-label'>🤖 Scrum Bot</div>
        <div class='bot-message'>{msg['content']}</div>
        """, unsafe_allow_html=True)

# ============================================================
# CHAT INPUT
# ============================================================
st.markdown("---")

default = st.session_state.pop("quick_q", "") if "quick_q" in st.session_state else ""
user_input = st.text_input("Type your question here:", value=default, placeholder="e.g. What is a Sprint Review?")

if st.button("Send 🚀"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    elif not user_input.strip():
        st.warning("Please type a question first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history[-6:]
                })
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style='text-align:center; color:#90a4ae; font-size:0.8rem; padding:1rem;'>
    Scrum Master Chat Bot • Powered by Scrum Guide 2020 • Built with LangChain + Groq + Streamlit
</div>
""", unsafe_allow_html=True)
