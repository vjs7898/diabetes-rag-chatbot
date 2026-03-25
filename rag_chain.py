from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# 1. Load the existing ChromaDB
print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 3. Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_completion_tokens=1024,
)

# 4. Custom prompt
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are an expert research assistant helping users understand a diabetes
classification research paper. Use ONLY the provided context to answer
questions. If the answer isn't in the context, say "I don't have enough
information in the paper to answer that."

Be specific — cite model names, accuracy numbers, and methodology details
from the paper when relevant.

Previous conversation:
{chat_history}

Relevant sections from the paper:
{context}

Question: {question}

Answer:"""
)

# 5. Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 6. Build RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
    return_source_documents=True,
    verbose=False
)

def ask(question: str) -> dict:
    result = qa_chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result["source_documents"]
    }

def show_sources(sources: list):
    print("\n--- Sources used ---")
    for i, doc in enumerate(sources, start=1):
        page = doc.metadata.get("page", "?")
        print(f"[{i}] Page {page}: {doc.page_content[:200].strip()}...")
    print("--------------------\n")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Diabetes Research Paper — RAG Chatbot")
    print("  Type 'quit' to exit | 'sources' to see last sources")
    print("=" * 55 + "\n")

    last_sources = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "sources":
            if last_sources:
                show_sources(last_sources)
            else:
                print("No sources yet — ask a question first.\n")
            continue

        print("Bot: ", end="", flush=True)
        result = ask(user_input)
        print(result["answer"])
        print()
        last_sources = result["sources"]