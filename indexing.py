import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
import qdrant_client

load_dotenv()

print("Starting indexing (with local model)...")

embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    trust_remote_code=True
)

node_parser = MarkdownNodeParser()

Settings.embed_model = embed_model
Settings.node_parser = node_parser

documents = SimpleDirectoryReader("./data/parsed").load_data()
print(f"Documents loaded: {len(documents)}")

client = qdrant_client.QdrantClient(url="http://localhost:6333")

try:
    client.delete_collection("innopolis_bot")
    print("Old/corrupted collection deleted.")
except:
    pass

vector_store = QdrantVectorStore(
    client=client,
    collection_name="innopolis_bot",
    enable_hybrid=True,
    batch_size=20
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Creating vectors...")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("Indexing completed! Database is ready.")