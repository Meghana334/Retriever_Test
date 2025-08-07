import cohere
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import re
from langchain.embeddings.base import Embeddings
from langchain_milvus import Milvus
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# 🔑 CONFIG & CLIENTS
# ----------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Initialize Cohere client
co = cohere.ClientV2(api_key=COHERE_API_KEY)




def clean_text(text):
    if not text or text == 'nan':
        return ""

    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
    words = text.split()
    cleaned_words = [word for word in words if len(word) > 2 or word.lower() in ['is', 'or', 'if', 'to', 'in', 'on', 'at']]

    result = ' '.join(cleaned_words)
    return result


def preprocess_query(query):
    """Preprocess query for better matching"""
    query = clean_text(query)
    query = query.replace("what", "").replace("how", "").replace("why", "").replace("when", "")
    query = query.replace("?", "").strip()
    return query

def chunk_pdf(pdf_path):
    print("Chunking PDF...")
    try:
        print("try")
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        print(pages)
        if not pages:
            logger.error("❌ PDF loaded but no pages found.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_documents(pages)
        print(chunks)
        cleaned_chunks = []
        skipped = 0

        for i, chunk in enumerate(chunks):
            print("in the loop")
            original = chunk.page_content
            if not original or original.strip() == "":
                logger.warning(f"[SKIPPED] Chunk {i+1} is empty.")
                skipped += 1
                continue

            cleaned = clean_text(original)
            print("**************************************************")
            print(f"[CHUNK {i+1}] Original: {len(original)} chars, Cleaned: {len(cleaned)} chars")
            print("**************************************************")
            print(cleaned)
            print("**************************************************")
            
            if len(cleaned.strip()) >= 10:
                chunk.page_content = cleaned
                cleaned_chunks.append(chunk)
            else:
                logger.warning(f"[SKIPPED] Chunk {i+1} too short after cleaning.")
                skipped += 1

            logger.info(f"[CHUNK {i+1}] {len(original)} → {len(cleaned)} chars")

        logger.info(f"✅ Final: {len(cleaned_chunks)} chunks | ❌ Skipped: {skipped}")
        return cleaned_chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return []


class CohereEmbeddings(Embeddings):
    def __init__(self, model: str = "embed-v4.0"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        # Cohere caps at 96 inputs per request:
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = [{"content": [{"type": "text", "text": text}]} for text in batch]
            resp = co.embed(
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
                inputs=inputs
            )
            all_embeddings.extend(resp.embeddings.float)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # single-input calls are fine
        resp = co.embed(
            model=self.model,
            input_type="search_query",
            embedding_types=["float"],
            texts=[text]
        )
        return resp.embeddings.float[0]