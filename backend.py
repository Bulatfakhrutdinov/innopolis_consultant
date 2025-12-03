import os
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank

class UniversityAgent:
    def __init__(self):
        load_dotenv()
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

        print("⚙️ Loading models...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-large",
            trust_remote_code=True
        )

        Settings.llm = Ollama(model="llama3", request_timeout=120.0, temperature=0.0)

        self.client = qdrant_client.QdrantClient(url="http://localhost:6333")
        self.vector_store = QdrantVectorStore(client=self.client, collection_name="innopolis_bot")
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        print("Loading Reranker...")
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=3
        )

        print("Agent initialized.")

    def get_chat_engine(self):
        system_prompt = (
            "You are an AI assistant for Innopolis University. "
            "Your task is to answer based strictly on the provided context documents. "

            "📢 LANGUAGE PROTOCOL (CRITICAL):"
            "1. IDENTIFY the language of the user's message."
            "2. ANSWER in the EXACT SAME language as the user."
            "   - If User speaks RUSSIAN -> You MUST answer in RUSSIAN. (Translate content if documents are in English)."
            "   - If User speaks ENGLISH -> You MUST answer in ENGLISH."

            "🚫 REFUSAL PROTOCOL (If answer is not in context):"
            "   - If you need to answer in RUSSIAN, say: 'К сожалению, в доступных документах информации об этом не найдено.'"
            "   - If you need to answer in ENGLISH, say: 'Unfortunately, information about this is not found in the documents.'"

            "✅ CONTENT RULES:"
            "1. Do not use outside knowledge. If it's not in the context, use the Refusal Protocol."
            "2. Be concise and professional."
        )

        return self.index.as_chat_engine(
            chat_mode="context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            node_postprocessors=[self.reranker],
            system_prompt=system_prompt
        )