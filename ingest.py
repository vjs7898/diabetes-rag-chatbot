from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── 1. Load the PDF ──────────────────────────────────────────────
print("Loading PDF...")
loader = PyPDFLoader("Masters_Project.pdf")
pages = loader.load()
print(f"  Loaded {len(pages)} pages")

# ── 2. Split into chunks ─────────────────────────────────────────
# chunk_size = how many characters per chunk
# chunk_overlap = how many characters shared between adjacent chunks
#   (overlap keeps context from getting cut off mid-sentence)
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]   # tries to split at paragraphs first
)
chunks = splitter.split_documents(pages)
print(f"  Created {len(chunks)} chunks from {len(pages)} pages")

# ── 3. Create embeddings (runs locally, no API cost) ─────────────
# This model converts each text chunk into a 384-dimension vector
# "all-MiniLM-L6-v2" is small, fast, and good enough for this project
print("Loading embedding model (downloads once, ~90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ── 4. Store vectors in ChromaDB ─────────────────────────────────
# persist_directory = folder where ChromaDB saves the vectors on disk
# This means you only run ingest.py ONCE — the vectors are saved
print("Storing vectors in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"  Saved {len(chunks)} vectors to ./chroma_db")

# ── 5. Quick retrieval test ───────────────────────────────────────
print("\nTesting retrieval...")
query = "What sampling strategy gave the best results?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 chunks for: '{query}'\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:300])
    print()

print("Ingestion complete! ChromaDB is ready.")