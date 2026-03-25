import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── All imports use modern langchain packages (0.2+/0.3+ safe) ────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Chatbot",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .source-box {
        background-color: #f0f4f8;
        border-left: 4px solid #1D9E75;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .badge {
        background: #E1F5EE;
        color: #0F6E56;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── API key check ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to your .env file locally, or in "
        "Settings → Variables and Secrets on Hugging Face Spaces."
    )
    st.stop()


# ── Cached resources (load once, reuse across reruns) ─────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Connecting to LLM...")
def load_llm():
    return ChatGroq(
        # USE this instead
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )


# ── PDF ingestion ─────────────────────────────────────────────────
def ingest_pdf(uploaded_file, embeddings):
    # PyPDFLoader needs a real file path, not bytes
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(pages)

    # In-memory vector store — no persist_directory needed
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore, len(pages), len(chunks)


# ── Build LCEL RAG chain ──────────────────────────────────────────
# Uses LangChain Expression Language — modern, works on all versions
def build_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant. Answer questions \
using ONLY the context provided from the research paper below. \
If the answer is not in the context, say clearly: \
"I don't have enough information in this paper to answer that." \
Be specific — cite model names, accuracy numbers, and methodology details.

Context from the paper:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL pipeline: retrieve → format → prompt → LLM → parse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(x["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Get answer + source docs ──────────────────────────────────────
def get_answer(chain, retriever, question, chat_history):
    source_docs = retriever.invoke(question)
    answer = chain.invoke({
        "question":     question,
        "chat_history": chat_history,
    })
    return answer, source_docs


# ── Session state defaults ────────────────────────────────────────
for key, default in {
    "messages":      [],   # list of {role, content} dicts for display
    "chat_history":  [],   # list of HumanMessage/AIMessage for LLM
    "chain":         None,
    "retriever":     None,
    "doc_stats":     None,
    "last_sources":  [],
    "uploaded_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Load cached resources ─────────────────────────────────────────
embeddings = load_embeddings()
llm        = load_llm()


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 Research Chatbot")
    st.caption("LangChain · ChromaDB · Groq (Llama 3) · Streamlit")
    st.divider()

    st.subheader("📄 Upload a Paper")
    uploaded_file = st.file_uploader(
        "Drop any research PDF",
        type="pdf",
        help="The PDF will be chunked and indexed automatically"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.uploaded_name:
            with st.spinner("Indexing PDF..."):
                try:
                    vectorstore, n_pages, n_chunks = ingest_pdf(
                        uploaded_file, embeddings
                    )
                    chain, retriever = build_chain(vectorstore, llm)

                    st.session_state.chain         = chain
                    st.session_state.retriever     = retriever
                    st.session_state.doc_stats     = {
                        "name":   uploaded_file.name,
                        "pages":  n_pages,
                        "chunks": n_chunks,
                    }
                    st.session_state.uploaded_name = uploaded_file.name
                    st.session_state.messages      = []
                    st.session_state.chat_history  = []
                    st.session_state.last_sources  = []
                    st.success("✅ Ready to chat!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Document Info")
        d = st.session_state.doc_stats
        st.markdown(f"**{d['name']}**")
        c1, c2 = st.columns(2)
        c1.metric("Pages",  d["pages"])
        c2.metric("Chunks", d["chunks"])

    st.divider()
    st.subheader("💡 Try asking...")
    suggestions = [
        "What models were compared?",
        "What was the best accuracy?",
        "How was class imbalance handled?",
        "What were the top predictive features?",
        "What are the study limitations?",
    ]
    for q in suggestions:
        if st.button(q, use_container_width=True, key=f"sq_{q[:15]}"):
            st.session_state["pending_q"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.rerun()


# ════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ════════════════════════════════════════════════════════════════
st.title("💬 Chat with your Research Paper")
st.caption("Upload any research PDF in the sidebar, then ask questions in plain English.")

if not st.session_state.chain:
    st.info("👈 Upload a research PDF in the sidebar to get started.")
    st.stop()

# ── Render chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Source citations expander ─────────────────────────────────────
if st.session_state.last_sources:
    with st.expander(
        f"📄 Sources used ({len(st.session_state.last_sources)} chunks)",
        expanded=False
    ):
        for i, doc in enumerate(st.session_state.last_sources):
            page    = doc.metadata.get("page", "?")
            snippet = doc.page_content.strip()[:300]
            st.markdown(
                f'<div class="source-box">'
                f'<span class="badge">Page {page}</span>'
                f'<span class="badge">Chunk {i+1}</span><br><br>'
                f'{snippet}...'
                f'</div>',
                unsafe_allow_html=True
            )

# ── Chat input ────────────────────────────────────────────────────
pending    = st.session_state.pop("pending_q", None)
user_input = st.chat_input("Ask anything about the paper...") or pending

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = get_answer(
                    st.session_state.chain,
                    st.session_state.retriever,
                    user_input,
                    st.session_state.chat_history,
                )
            except Exception as e:
                answer  = f"Sorry, something went wrong: {str(e)}"
                sources = []

        # Simulate token streaming word by word
        placeholder = st.empty()
        streamed    = ""
        for word in answer.split():
            streamed += word + " "
            placeholder.markdown(streamed + "▌")
        placeholder.markdown(answer)

    # Save to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Update LLM chat history (keep last 20 messages = 10 exchanges)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.chat_history = st.session_state.chat_history[-20:]

    st.session_state.last_sources = sources
    st.rerun()