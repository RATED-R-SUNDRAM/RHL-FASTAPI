# ingest_lcc.py

import os
import json
import time
import hashlib
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
#from unstructured.partition.pdf import partition_pdf
#from unstructured.partition.docx import partition_docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI   # works for both Anthropic + OpenAI-compatible APIs

# ---------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------
load_dotenv()
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY_NEW"))
index_name = "medical-chatbot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # set in .env

print("passed all imports ........")
# ---------------------------------------------------------------------
# 2. Extraction
# ---------------------------------------------------------------------
import fitz  # add at top

def extract_elements(filepath):
    docs = []
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        print(f"..... at {filepath}")
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                docs.append({
                    "text": text,
                    "type": "page",
                    "metadata": {"page": i+1}
                })
    elif filepath.endswith(".docx"):
        from unstructured.partition.docx import partition_docx
        elements = partition_docx(filepath)
        for el in elements:
            if el.text.strip():
                docs.append({
                    "text": el.text,
                    "type": el.category,
                    "metadata": {}
                })
    else:
        raise ValueError("Unsupported file type")
    return docs


# ---------------------------------------------------------------------
# 3. Splitting
# ---------------------------------------------------------------------
def split_documents(docs, chunk_size=600, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk": chunk,
                "metadata": {**doc["metadata"], "type": doc["type"], "chunk_id": i}
            })
    return all_chunks


# ---------------------------------------------------------------------
# 4. Anthropic-Style LCC Rewriting
# ---------------------------------------------------------------------
def rewrite_chunk_with_context(chunks, window=1):
    rewritten_chunks = []
    for i, c in enumerate(chunks):
        neighbors = []
        if i - window >= 0:
            neighbors.append(chunks[i - window]["chunk"])
        if i + window < len(chunks):
            neighbors.append(chunks[i + window]["chunk"])

        context_text = "\n".join(neighbors)
        prompt = f"""
You are an agent who is expert in chunking by taking a particular chunk augomenting with local context so that chunk has individual meaning of its own.add()
For eg . 

Context_text  : Jaundice is a disease , it is in liver 
chunk  : It is a dangerous disease

rewritten chunk : Jaundice is a dangerous disease in the liver

Chunk:
{c["chunk"]}

Neighboring context:
{context_text}

Rewritten, self-contained chunk:
"""
        # LLM call (Anthropic Claude or OpenAI GPT-4o, depending on API key)
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",  # swap with Anthropic Claude if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        rewritten = resp.choices[0].message.content.strip()
        rewritten_chunks.append({
            "chunk": rewritten,
            "metadata": c["metadata"]
        })

        # avoid hammering API
        time.sleep(0.3)
    return rewritten_chunks


# ---------------------------------------------------------------------
# 5. Embedding + Pinecone
# ---------------------------------------------------------------------
def batch_embed_upsert(chunks, batch_size=32):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["chunk"] for c in batch]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)

        vectors = []
        for j, (text, emb) in enumerate(zip(texts, embeddings)):
            uid = hashlib.md5(text.encode()).hexdigest()
            vectors.append({
                "id": uid,
                "values": emb.tolist(),
                "metadata": {
                    "text": text,
                    **batch[j]["metadata"]
                }
            })

        index.upsert(vectors)
        print(f"Upserted {len(vectors)} vectors")


# ---------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------
def process_file(filepath):
    docs = extract_elements(filepath)
    base_chunks = split_documents(docs)
    lcc_chunks = rewrite_chunk_with_context(base_chunks, window=1)
    batch_embed_upsert(lcc_chunks)


if __name__ == "__main__":
    folder = r"D:\Documents\RHL-RAG-PROJECT\FILES"
    for file in os.listdir(folder):
        if file.endswith((".pdf", ".docx")):
            print(f"processing {file}....")
            process_file(os.path.join(folder, file))
