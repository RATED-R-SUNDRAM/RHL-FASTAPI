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
from pdf2image import convert_from_path


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
