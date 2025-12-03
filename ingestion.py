import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

nest_asyncio.apply()
load_dotenv()

if not os.getenv("LLAMA_CLOUD_API_KEY"):
    raise ValueError("Please add LLAMA_CLOUD_API_KEY to the .env file")

print("Starting document parsing...")

parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="ru",
    parsing_instruction="""
    This document contains academic rules, curricula, and tables from Innopolis University in Russian.
    Please represent tables strictly as Markdown tables.
    Do not lose headings and row information.
    Preserve the original text in Russian, do not translate it.
    """
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    "./data/raw",
    file_extractor=file_extractor
).load_data()

print(f"Successfully processed {len(documents)} fragments/pages.")

output_path = "./data/parsed"
os.makedirs(output_path, exist_ok=True)

doc_text_map = {}

for doc in documents:
    file_name = os.path.basename(doc.metadata["file_path"]).replace(".pdf", ".md")
    if file_name not in doc_text_map:
        doc_text_map[file_name] = ""
    doc_text_map[file_name] += doc.text + "\n\n"

for file_name, content in doc_text_map.items():
    with open(f"{output_path}/{file_name}", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved file: {file_name}")

print("\nParsing completed! Check the data/parsed folder")