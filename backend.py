import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank


class UniversityAgent:
    def __init__(self):
        load_dotenv()
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

        print("Loading models...")

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-large",
            trust_remote_code=True
        )

        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        print(f"🦙 Connecting to Ollama at: {ollama_url}")

        Settings.llm = Ollama(
            model="llama3",
            base_url=ollama_url,
            request_timeout=120.0,
            temperature=0.0
        )
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        print(f"🔌 Connecting to Qdrant at: {qdrant_url}")

        self.client = QdrantClient(url=qdrant_url)
        self.aclient = AsyncQdrantClient(url=qdrant_url)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name="innopolis_bot"
        )

        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        print("Loading Reranker...")
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=3
        )

        print("✅ Agent initialized successfully.")

    def get_chat_engine(self):
        system_prompt = (
            "You are an AI assistant for Innopolis University. "

            "👋 GREETING PROTOCOL (PRIORITY 1):"
            "If the user input is a greeting (e.g., 'Hello', 'Hi', 'Привет', 'Здравствуйте'), "
            "IGNORE the context documents. Reply politely in the user's language and offer help. "
            "Do NOT say 'I cannot find information' for greetings."

            "📢 LANGUAGE PROTOCOL (PRIORITY 2):"
            "1. IDENTIFY the language of the user's message."
            "2. ANSWER in the EXACT SAME language as the user."
            "   - If User speaks RUSSIAN -> You MUST answer in RUSSIAN. (Translate content if documents are in English)."
            "   - If User speaks ENGLISH -> You MUST answer in ENGLISH."

            "🚫 REFUSAL PROTOCOL (For questions only):"
            "   - If it is a QUESTION and the answer is NOT in the context:"
            "     * In Russian: 'К сожалению, в доступных документах информации об этом не найдено.'"
            "     * In English: 'Unfortunately, information about this is not found in the documents.'"

            "✅ CONTENT RULES:"
            "1. Answer strictly based on the provided context."
            "2. Be concise and professional."
        )

        return self.index.as_chat_engine(
            chat_mode="context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            node_postprocessors=[self.reranker],
            system_prompt=system_prompt
        )