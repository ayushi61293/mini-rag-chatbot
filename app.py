import os
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_db = None
embeddings = None
llm = None


@app.on_event("startup")
async def load_models():
    global embeddings, llm
    try:
        print("⏳ Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
        )
        print("✅ Embeddings loaded")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key,
        )
        print("✅ Groq Llama3 ready!")

    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise e


@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running!"}


@app.get("/health")
def health():
    return {
        "embeddings": embeddings is not None,
        "llm": llm is not None,
        "vector_db": vector_db is not None,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        if embeddings is None:
            raise HTTPException(status_code=503, detail="Models still loading. Please wait.")

        tmp_path = "temp_uploaded.pdf"
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        print(f"📄 PDF saved: {file.filename}")

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(docs, embeddings)
        print(f"✅ Indexed {len(docs)} chunks")

        return {"status": "ok", "pages": len(documents), "chunks": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat history message model ──
class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []   # full conversation history
    show_source: bool = False


TECH_MAP = {
    "python": "Python", "fastapi": "FastAPI", "kafka": "Kafka",
    "redis": "Redis", "mongodb": "MongoDB", "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL", "whisper": "Whisper", "deepgram": "Deepgram",
    "huggingface": "HuggingFace Transformers", "transformers": "HuggingFace Transformers",
    "openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Gemini",
    "langchain": "LangChain", "faiss": "FAISS", "pinecone": "Pinecone",
    "docker": "Docker", "github": "GitHub Actions", "linux": "Linux",
}


def extract_technologies(docs):
    found = set()
    for doc in docs:
        text = doc.page_content.lower()
        for key, value in TECH_MAP.items():
            if key in text:
                found.add(value)
    return sorted(found)


def clean_text(text):
    return text.replace("\n", " ").replace("DISCLAIMER", "").replace("\ufb01", "fi")


def build_messages(context: str, query: str, history: List[ChatMessage]):
    """Build message list with system prompt + chat history + new question."""
    messages = [
        SystemMessage(content=(
            "You are a helpful AI assistant that answers questions based strictly on the provided document context. "
            "You also remember the conversation history and can answer follow-up questions. "
            "Give clear, accurate, and complete answers. "
            "If the answer is not in the context, say: 'This information is not available in the document.'"
        ))
    ]

    # Add last 6 messages of history (3 exchanges) to keep context window small
    recent_history = history[-6:] if len(history) > 6 else history
    for msg in recent_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    # Add current question with document context
    messages.append(HumanMessage(content=(
        f"Document context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )))

    return messages


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        if vector_db is None:
            raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not ready yet.")

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )
        retrieved = retriever.invoke(req.query)
        print(f"🔍 Query: {req.query} | History: {len(req.history)} messages")

        is_tech = any(
            w in req.query.lower()
            for w in ["technology", "technologies", "tools", "stack", "framework", "tech"]
        )

        if is_tech:
            techs = extract_technologies(retrieved)
            answer = "\n".join(f"- {t}" for t in techs) if techs else "No technologies found."
        else:
            context = "\n\n".join(clean_text(d.page_content) for d in retrieved[:5])
            messages = build_messages(context, req.query, req.history)
            response = llm.invoke(messages)
            answer = response.content
            print(f"🤖 Answer: {answer[:100]}...")

        sources = []
        if req.show_source:
            sources = [clean_text(d.page_content[:200]) + "…" for d in retrieved]

        return {"answer": answer, "sources": sources}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
