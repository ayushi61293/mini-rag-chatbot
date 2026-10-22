import os
import re
import tempfile
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

# Single shared index (demo design). For multi-user use, keep one index per session.
vector_db = None
all_docs = []          # all chunks, used by the tech-extraction shortcut
embeddings = None
llm = None

TOP_K = 5              # one k everywhere: what the model reads == what sources show
HISTORY_MESSAGES = 6   # last 3 exchanges


@app.on_event("startup")
async def load_models():
    global embeddings, llm
    try:
        print(" Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(" Embeddings loaded")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key,
        )
        print(" Groq Llama3 ready!")

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
    global vector_db, all_docs
    tmp_path = None
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        if embeddings is None:
            raise HTTPException(status_code=503, detail="Models still loading. Please wait.")

        contents = await file.read()

        # Unique temp file per request, so concurrent uploads don't collide
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        print(f"📄 PDF saved: {file.filename}")

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(docs, embeddings)
        all_docs = docs
        print(f" Indexed {len(docs)} chunks")

        return {"status": "ok", "pages": len(documents), "chunks": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        print(f" Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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

# Word-boundary match; "tech" alone is removed so "technical approach" no longer triggers it
TECH_QUERY_PATTERN = re.compile(
    r"\b(technology|technologies|tools|tech stack|stack|frameworks?)\b"
)


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


def rewrite_query(query: str, history: List[ChatMessage]) -> str:
    """Turn a follow-up into a standalone question using chat history.
    Used ONLY for retrieval; the original question still goes to the LLM."""
    if not history:
        return query
    try:
        convo = "\n".join(f"{m.role}: {m.content}" for m in history[-HISTORY_MESSAGES:])
        prompt = (
            "Given the conversation and a follow-up question, rewrite the follow-up "
            "as a standalone question that includes all needed context. "
            "Do NOT answer it. If it is already standalone, return it unchanged. "
            "Output only the question.\n\n"
            f"Conversation:\n{convo}\n\n"
            f"Follow-up: {query}\n\nStandalone question:"
        )
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or query
    except Exception as e:
        print(f"⚠️ Rewrite failed, using original query: {e}")
        return query


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
    recent_history = history[-HISTORY_MESSAGES:]
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

        # 1. Rewrite follow-ups into standalone queries for retrieval
        search_query = rewrite_query(req.query, req.history)

        # 2. Retrieve with the rewritten query
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        retrieved = retriever.invoke(search_query)
        print(f"🔍 Original: {req.query} | Rewritten: {search_query} | History: {len(req.history)} msgs")

        is_tech = bool(TECH_QUERY_PATTERN.search(req.query.lower()))

        if is_tech:
            # Scan ALL chunks, not just the retrieved ones
            techs = extract_technologies(all_docs)
            answer = "\n".join(f"- {t}" for t in techs) if techs else "No technologies found."
        else:
            context = "\n\n".join(clean_text(d.page_content) for d in retrieved)
            messages = build_messages(context, req.query, req.history)
            response = llm.invoke(messages)
            answer = response.content
            print(f"🤖 Answer: {answer[:100]}...")

        sources = []
        if req.show_source:
            sources = [clean_text(d.page_content[:200]) + "…" for d in retrieved]

        return {"answer": answer, "sources": sources, "search_query": search_query}

    except HTTPException:
        raise
    except Exception as e:
        print(f" Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
