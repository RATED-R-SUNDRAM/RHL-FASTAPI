# ingest_to_pinecone.py
import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

# Extraction / parsing
import fitz                         # PyMuPDF for PDFs
from docx import Document           # python-docx for DOCX

# Chunking + embedding
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone (new SDK)
from pinecone import Pinecone, ServerlessSpec

# Load .env early
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")

# Config
PINECONE_KEY = os.getenv("PINECONE_API_KEY_NEW")
if not PINECONE_KEY:
    logging.error("PINECONE_API_KEY_NEW not set in environment. Exiting.")
    sys.exit(1)

# Embedding model (all-mpnet-base-v2)
logging.info("Loading embedding model (this can take time)...")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()
logging.info(f"Embedding dim = {EMBED_DIM}")

# Pinecone client + index
pc = Pinecone(api_key=PINECONE_KEY)
INDEX_NAME = "medical-chatbot-index"

# fetch existing index names defensively
try:
    existing = pc.list_indexes().names()
except Exception:
    try:
        existing = pc.list_indexes()
    except Exception as e:
        logging.error("Unable to list Pinecone indexes: %s", e)
        raise

if INDEX_NAME not in existing:
    logging.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM})...")
    pc.create_index(
        name=INDEX_NAME,
        metric="cosine",
        dimension=EMBED_DIM,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)
logging.info("Pinecone index ready.")

def extract_from_pdf(path: str):
    """
    Return list of element dicts: {'text','type','page','section'}
    Heuristic: treat block as heading if max span size >> median page font size.
    """
    doc = fitz.open(path)
    elements = []
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        # collect font sizes on page
        sizes = []
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span.get("size", 0))
        median_font = float(np.median(sizes)) if sizes else 0.0

        current_section = None
        for b in blocks:
            # assemble block text
            spans_text = []
            max_span_size = 0.0
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        spans_text.append(text)
                        max_span_size = max(max_span_size, span.get("size", 0.0))
            block_text = " ".join(spans_text).strip()
            if not block_text:
                continue
            # header heuristic
            is_header = False
            if median_font and max_span_size >= median_font * 1.15 and len(block_text.split()) <= 12:
                is_header = True

            if is_header:
                current_section = block_text
                elements.append({"text": block_text, "type": "heading", "page": pno, "section": current_section})
            else:
                elements.append({"text": block_text, "type": "paragraph", "page": pno, "section": current_section})
    return elements


def extract_from_docx(path: str):
    """
    Return list of element dicts from DOCX: rely on styles ("Heading") to detect sections.
    """
    doc = Document(path)
    elements = []
    current_section = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = getattr(para, "style", None)
        style_name = getattr(style, "name", "") if style else ""
        # consider headings
        if style_name and style_name.lower().startswith("heading"):
            current_section = text
            elements.append({"text": text, "type": "heading", "page": None, "section": current_section})
        else:
            elements.append({"text": text, "type": "paragraph", "page": None, "section": current_section})
    return elements


def parse_document(filepath: str):
    filepath = str(filepath)
    if filepath.lower().endswith(".pdf"):
        return extract_from_pdf(filepath)
    elif filepath.lower().endswith(".docx"):
        return extract_from_docx(filepath)
    else:
        raise ValueError("Unsupported file type: " + filepath)

def make_chunks(elements, doc_name: str, chunk_size: int = 700, chunk_overlap: int = 120):
    """
    Combine paragraph texts into larger chunks (~700 words with overlap).
    Section/page metadata is carried into each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Collect all text into one string with markers
    docs_texts = []
    metas = []
    buffer = []
    last_meta = None

    for el_idx, el in enumerate(elements):
        text = el["text"].strip()
        if not text:
            continue

        # If it's a heading, flush buffer and reset
        if el["type"] == "heading":
            if buffer:
                docs_texts.append(" ".join(buffer))
                metas.append(last_meta)
                buffer = []
            last_meta = {
                "doc_name": doc_name,
                "page": el.get("page"),
                "section": text,
                "type": el.get("type"),
                "el_index": el_idx,
            }
        else:
            buffer.append(text)
            last_meta = {
                "doc_name": doc_name,
                "page": el.get("page"),
                "section": el.get("section"),
                "type": el.get("type"),
                "el_index": el_idx,
            }

    # Flush last buffer
    if buffer:
        docs_texts.append(" ".join(buffer))
        metas.append(last_meta)

    # Now split into ~700 word chunks
    chunks_out = []
    for meta, big_text in zip(metas, docs_texts):
        subchunks = splitter.split_text(big_text)
        for i, sc in enumerate(subchunks):
            meta_copy = dict(meta)
            meta_copy["chunk_index"] = i
            chunks_out.append({"text": sc, "metadata": meta_copy})

    return chunks_out

def batch_embed_upsert(chunks, batch_size: int = 32):
    if not chunks:
        logging.warning("No chunks to upsert.")
        return

    total = len(chunks)
    logging.info(f"Embedding & upserting {total} chunks (batch_size={batch_size})...")
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        # embed
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)

        vectors = []
        for j, (text, emb) in enumerate(zip(texts, embeddings)):
            meta = clean_metadata(batch[j]["metadata"])   # sanitize metadata
            uid_source = f"{meta.get('doc_name','')}_{meta.get('el_index')}_{meta.get('chunk_index')}"
            vid = hashlib.md5(uid_source.encode("utf-8")).hexdigest()

            vectors.append({
                "id": vid,
                "values": emb.tolist(),
                "metadata": {
                    **meta,
                    "text_full": text,            # ✅ full chunk goes here
                    "text_snippet": text[:400]    # ✅ short preview
                }
            })

        try:
            index.upsert(vectors)
            logging.info(f"Upserted batch {(i//batch_size)+1} ({len(vectors)} vectors).")
        except Exception as e:
            logging.error("Failed upsert (batch %d): %s", i//batch_size+1, e)
            raise
        time.sleep(0.1)  # slight pause
def clean_metadata(meta: dict) -> dict:
    safe_meta = {}
    for k, v in meta.items():
        if v is None:
            safe_meta[k] = "unknown"   # 👈 force string instead of None
        elif isinstance(v, (str, int, float, bool)):
            safe_meta[k] = v
        elif isinstance(v, list):
            safe_meta[k] = [str(x) for x in v if x is not None]
        else:
            safe_meta[k] = str(v)
    return safe_meta

def process_file(path: str):
    logging.info(f"Processing file: {path}")
    elements = parse_document(path)
    logging.info(f"Extracted {len(elements)} elements from {os.path.basename(path)}")
    chunks = make_chunks(elements, doc_name=os.path.basename(path))
    logging.info(f"Generated {len(chunks)} chunks from {os.path.basename(path)}")
    if chunks:
        # print a couple of samples
        for s in chunks[:2]:
            logging.debug("SAMPLE CHUNK META: %s", json.dumps(s["metadata"], default=str))
            logging.debug("SAMPLE CHUNK TEXT (start): %s", s["text"][:160].replace("\n"," "))
        batch_embed_upsert(chunks)
    else:
        logging.warning("No chunks for file %s", path)


def process_folder(folder_path: str):
    p = Path(folder_path)
    if not p.exists():
        logging.error("Folder does not exist: %s", folder_path)
        return
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in (".pdf", ".docx"):
            process_file(str(f))
if __name__ == "__main__":
    FOLDER = r"D:\Documents\RHL-RAG-PROJECT\FILES"   # <-- change this
    logging.info("Starting ingestion...")
    process_folder(FOLDER)
    logging.info("Ingestion complete.")
