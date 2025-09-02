# Imports
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
# 1. Embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 2. Pinecone initialization (new SDK)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY_NEW"))

# 3. Create or connect to index
index_name = "medical-chatbot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,   # mpnet-base-v2 output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

print("HERE")
# 4. Quick sanity check
sample_text = "WHO guidelines for managing preeclampsia"
embedding_vector = embedding_model.encode(sample_text, convert_to_numpy=True)
print("Embedding vector shape:", embedding_vector.shape)  # (768,)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # keeps chunks semantic + embedding friendly
    chunk_overlap=100
)

def parse_document(filepath):
    """Parse PDF/DOCX using Unstructured to preserve layout info."""
    if filepath.endswith(".pdf"):
        elements = partition_pdf(filename=filepath, strategy="hi_res")
    elif filepath.endswith(".docx"):
        elements = partition_docx(filename=filepath)
    else:
        raise ValueError("Unsupported file type.")
    return elements


def process_elements(elements, doc_name):
    """Convert unstructured elements → semantic chunks with metadata."""
    docs = []

    for i, el in enumerate(elements):
        el_text = el.text.strip()
        if not el_text:
            continue

        # Split into smaller semantic chunks
        chunks = splitter.split_text(el_text)

        for j, chunk in enumerate(chunks):
            metadata = {
                "doc_name": doc_name,
                "section_type": el.category,  # paragraph, title, table, list, etc.
                "page_num": getattr(el.metadata, "page_number", None),
                "chunk_id": f"{i}-{j}",
            }

            docs.append({"text": chunk, "metadata": metadata})

    return docs


def embed_and_push(docs):
    """Embed and upsert into Pinecone with metadata."""
    vectors = []
    for d in docs:
        emb = embedding_model.encode(d["text"]).tolist()
        vec_id = f'{d["metadata"]["doc_name"]}-{d["metadata"]["chunk_id"]}'
        vectors.append(
            {
                "id": vec_id,
                "values": emb,
                "metadata": {**d["metadata"], "text": d["text"]}
            }
        )
    
    index.upsert(vectors)


# --------------------------
# Main Loop (overnight run)
# --------------------------

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not (filename.endswith(".pdf") or filename.endswith(".docx")):
            continue

        print(f"Processing {filename}...")
        elements = parse_document(filepath)
        docs = process_elements(elements, doc_name=filename)
        embed_and_push(docs)
        print(f"✅ {filename} processed and pushed to Pinecone.")


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    folder = r"D:\Documents\RHL-RAG-PROJECT\FILES"  # change to your folder path
    process_folder(folder)
