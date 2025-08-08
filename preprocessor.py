# import cohere
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# import logging
# import re
# from langchain.embeddings.base import Embeddings
# from langchain_milvus import Milvus
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# import os
# import json
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# load_dotenv()


# # ----------------------------
# # 🔑 CONFIG & CLIENTS
# # ----------------------------
# COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
# MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# # Initialize Cohere client
# co = cohere.ClientV2(api_key=COHERE_API_KEY)




# def clean_text(text):
#     if not text or text == 'nan':
#         return ""

#     text = str(text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
#     words = text.split()
#     cleaned_words = [word for word in words if len(word) > 2 or word.lower() in ['is', 'or', 'if', 'to', 'in', 'on', 'at']]

#     result = ' '.join(cleaned_words)
#     return result


# def preprocess_query(query):
#     """Preprocess query for better matching"""
#     query = clean_text(query)
#     query = query.replace("what", "").replace("how", "").replace("why", "").replace("when", "")
#     query = query.replace("?", "").strip()
#     return query

# def chunk_pdf(pdf_path):
#     print("Chunking PDF...")
#     try:
#         print("try")
#         loader = PyMuPDFLoader(pdf_path)
#         pages = loader.load()
#         print(pages)
#         if not pages:
#             logger.error("❌ PDF loaded but no pages found.")
#             return []

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
#             length_function=len,
#         )

#         chunks = splitter.split_documents(pages)
#         print(chunks)
#         cleaned_chunks = []
#         skipped = 0

#         for i, chunk in enumerate(chunks):
#             print("in the loop")
#             original = chunk.page_content
#             if not original or original.strip() == "":
#                 logger.warning(f"[SKIPPED] Chunk {i+1} is empty.")
#                 skipped += 1
#                 continue

#             cleaned = clean_text(original)
#             print("**************************************************")
#             print(f"[CHUNK {i+1}] Original: {len(original)} chars, Cleaned: {len(cleaned)} chars")
#             print("**************************************************")
#             print(cleaned)
#             print("**************************************************")
            
#             if len(cleaned.strip()) >= 10:
#                 chunk.page_content = cleaned
#                 chunk.metadata["chunk"] = i
#                 cleaned_chunks.append(chunk)
#             else:
#                 logger.warning(f"[SKIPPED] Chunk {i+1} too short after cleaning.")
#                 skipped += 1

#             logger.info(f"[CHUNK {i+1}] {len(original)} → {len(cleaned)} chars")

#         logger.info(f"✅ Final: {len(cleaned_chunks)} chunks | ❌ Skipped: {skipped}")
#         return cleaned_chunks

#     except Exception as e:
#         logger.error(f"Chunking failed: {e}")
#         return []


# class CohereEmbeddings(Embeddings):
#     def __init__(self, model: str = "embed-v4.0"):
#         self.model = model

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         all_embeddings = []
#         # Cohere caps at 96 inputs per request:
#         batch_size = 96
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i : i + batch_size]
#             inputs = [{"content": [{"type": "text", "text": text}]} for text in batch]
#             resp = co.embed(
#                 model=self.model,
#                 input_type="search_document",
#                 embedding_types=["float"],
#                 inputs=inputs
#             )
#             all_embeddings.extend(resp.embeddings.float)
#         return all_embeddings

#     def embed_query(self, text: str) -> List[float]:
#         # single-input calls are fine
#         resp = co.embed(
#             model=self.model,
#             input_type="search_query",
#             embedding_types=["float"],
#             texts=[text]
#         )
#         return resp.embeddings.float[0]


import os
import logging
import math
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Core imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import cohere
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CohereEmbeddings:
    """Custom Cohere embeddings wrapper"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
                truncate="END"
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
                truncate="END"
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []


class BM25Retriever:
    """BM25 implementation for sparse retrieval"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_frequencies = {}
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0
        
    def fit(self, documents: List[str]):
        """Fit the BM25 model on documents"""
        self.documents = documents
        self.N = len(documents)
        
        # Calculate document frequencies and lengths
        self.doc_lengths = []
        word_doc_freq = defaultdict(int)
        
        for doc in documents:
            words = self._tokenize(doc)
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                word_doc_freq[word] += 1
        
        self.doc_frequencies = dict(word_doc_freq)
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_term_frequency(self, term: str, document: str) -> int:
        """Get term frequency in document"""
        words = self._tokenize(document)
        return words.count(term)
    
    def _get_idf(self, term: str) -> float:
        """Calculate IDF score"""
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5))
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        query_terms = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            score = 0
            doc_length = self.doc_lengths[i]
            
            for term in query_terms:
                tf = self._get_term_frequency(term, doc)
                idf = self._get_idf(term)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class DocumentProcessor:
    """Document loading and preprocessing utilities"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and chunk PDF document"""
        logger.info(f"Loading PDF from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
        
        # Load PDF pages
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        # Split into chunks
        documents = self.splitter.split_documents(pages)
        logger.info(f"Split PDF into {len(documents)} chunks")
        
        return documents
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess documents (can be extended for more complex preprocessing)"""
        processed_docs = []
        
        for doc in documents:
            # Basic text cleaning
            content = doc.page_content.strip()
            
            # Skip very short chunks
            if len(content) < 50:
                continue
                
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Update document
            doc.page_content = content
            processed_docs.append(doc)
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs


class ReRanker:
    """Cross-encoder re-ranking utilities"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        logger.info("Loading cross-encoder model for re-ranking...")
        self.cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")
        
        # Initialize sentence transformer for similarity scoring
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def rerank_documents(self, query: str, candidate_docs: List[Document]) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not candidate_docs:
            return candidate_docs
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for doc in candidate_docs:
            # Truncate document content if too long (cross-encoder has token limits)
            doc_text = doc.page_content[:512]  # Limit to 512 characters
            query_doc_pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        try:
            cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Create list of (document, score) pairs
            doc_score_pairs = list(zip(candidate_docs, cross_encoder_scores))
            
            # Sort by cross-encoder score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return sorted documents
            reranked_docs = [doc for doc, score in doc_score_pairs]
            
            logger.info(f"Cross-encoder re-ranking completed. Top scores: {cross_encoder_scores[:3]}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during cross-encoder re-ranking: {e}")
            return candidate_docs
    
    def calculate_similarity_score(self, answer: str, ground_truth: str) -> float:
        """Calculate cosine similarity between answer and ground truth"""
        if not answer or not ground_truth:
            return 0.0
        
        try:
            answer_emb = self.sentence_transformer.encode(answer)
            gt_emb = self.sentence_transformer.encode(ground_truth)
            similarity = cosine_similarity([answer_emb], [gt_emb])[0][0]
            return max(0.0, float(similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


def reciprocal_rank_fusion(dense_results: List[Tuple[int, float]], 
                          sparse_results: List[Tuple[int, float]], 
                          k: int = 60) -> List[Tuple[int, float]]:
    """Implement Reciprocal Rank Fusion"""
    
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    
    # Process dense results
    for rank, (doc_idx, score) in enumerate(dense_results):
        rrf_scores[doc_idx] += 1.0 / (rank + k)
    
    # Process sparse results
    for rank, (doc_idx, score) in enumerate(sparse_results):
        rrf_scores[doc_idx] += 1.0 / (rank + k)
    
    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results