from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load your API key from .env
load_dotenv()

# Initialize the LLM — we're using Llama 3 on Groq (free & fast)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
)

# Send a test message
response = llm.invoke("In one sentence, what is Retrieval Augmented Generation?")
print(response.content)