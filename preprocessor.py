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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Some features may be limited.")

logger = logging.getLogger(__name__)


class CohereEmbeddings:
    """Custom Cohere embeddings wrapper with proper error handling"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
        logger.info(f"Initialized Cohere embeddings with model: {model}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with batch processing"""
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts after filtering")
                return []
            
            # Cohere has a limit of 96 texts per request
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}")
                
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document",
                    truncate="END"
                )
                all_embeddings.extend(response.embeddings)
            
            logger.info(f"Successfully embedded {len(valid_texts)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 1024 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not text or not text.strip():
            logger.warning("Empty query provided for embedding")
            return [0.0] * 1024
        
        try:
            response = self.client.embed(
                texts=[text.strip()],
                model=self.model,
                input_type="search_query",
                truncate="END"
            )
            return response.embeddings[0]
            
        except Exception as e:
            logger.error(f"Error embedding query '{text[:50]}...': {e}")
            return [0.0] * 1024


class BM25Retriever:
    """BM25 implementation for sparse retrieval with proper tokenization"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_frequencies = {}
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0
        logger.info(f"Initialized BM25 with k1={k1}, b={b}")
        
    def fit(self, documents: List[str]):
        """Fit the BM25 model on documents"""
        if not documents:
            logger.warning("No documents provided for BM25 fitting")
            return
        
        logger.info(f"Fitting BM25 on {len(documents)} documents...")
        
        self.documents = documents
        self.N = len(documents)
        
        # Calculate document frequencies and lengths
        self.doc_lengths = []
        word_doc_freq = defaultdict(int)
        
        for i, doc in enumerate(documents):
            if not doc:
                logger.warning(f"Empty document at index {i}")
                self.doc_lengths.append(0)
                continue
                
            words = self._tokenize(doc)
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                word_doc_freq[word] += 1
        
        self.doc_frequencies = dict(word_doc_freq)
        
        # Calculate average document length (avoid division by zero)
        valid_lengths = [length for length in self.doc_lengths if length > 0]
        self.avg_doc_length = sum(valid_lengths) / len(valid_lengths) if valid_lengths else 1.0
        
        logger.info(f"BM25 fitting completed. Vocabulary size: {len(self.doc_frequencies)}")
        logger.info(f"Average document length: {self.avg_doc_length:.2f}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with preprocessing"""
        if not text:
            return []
        
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove extra whitespace and split on word boundaries
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out very short words (optional)
        words = [word for word in words if len(word) >= 2]
        
        return words
    
    def _get_term_frequency(self, term: str, document: str) -> int:
        """Get term frequency in document"""
        if not document:
            return 0
        words = self._tokenize(document)
        return words.count(term)
    
    def _get_idf(self, term: str) -> float:
        """Calculate IDF score with smoothing"""
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        
        # Add smoothing to prevent negative IDF
        idf = math.log((self.N - df + 0.5) / (df + 0.5))
        return max(0.0, idf)  # Ensure non-negative IDF
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        if not query or not query.strip():
            logger.warning("Empty query provided for BM25 scoring")
            return [0.0] * len(self.documents)
        
        if not self.documents:
            logger.warning("No documents available for BM25 scoring")
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            logger.warning(f"No valid terms extracted from query: '{query}'")
            return [0.0] * len(self.documents)
        
        scores = []
        
        for i, doc in enumerate(self.documents):
            if not doc:
                scores.append(0.0)
                continue
                
            score = 0.0
            doc_length = self.doc_lengths[i]
            
            # Avoid division by zero
            if doc_length == 0:
                scores.append(0.0)
                continue
            
            for term in query_terms:
                tf = self._get_term_frequency(term, doc)
                idf = self._get_idf(term)
                
                if tf > 0 and idf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores
    
    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k documents for a query"""
        scores = self.get_scores(query)
        
        # Create (doc_index, score) pairs and sort by score
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:k]


class DocumentProcessor:
    """Document loading and preprocessing utilities with enhanced chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and chunk PDF document with enhanced error handling"""
        logger.info(f"Loading PDF from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
        
        try:
            # Load PDF pages
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                raise ValueError(f"No pages loaded from PDF: {pdf_path}")
            
            logger.info(f"Loaded {len(pages)} pages from PDF")
            
            # Split into chunks
            documents = self.splitter.split_documents(pages)
            logger.info(f"Split PDF into {len(documents)} chunks")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        
        # Remove very short words except common ones
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(word) > 2 or word.lower() in ['is', 'or', 'if', 'to', 'in', 'on', 'at', 'a', 'an']:
                cleaned_words.append(word)
        
        result = ' '.join(cleaned_words)
        return result.strip()
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess documents with cleaning and filtering"""
        if not documents:
            logger.warning("No documents provided for preprocessing")
            return []
        
        logger.info(f"Preprocessing {len(documents)} documents...")
        
        processed_docs = []
        skipped_count = 0
        
        for i, doc in enumerate(documents):
            # Clean the content
            original_content = doc.page_content
            cleaned_content = self.clean_text(original_content)
            
            # Skip very short chunks
            if len(cleaned_content) < 50:
                logger.debug(f"Skipping short document {i}: {len(cleaned_content)} chars")
                skipped_count += 1
                continue
            
            # Update document content
            doc.page_content = cleaned_content
            
            # Ensure metadata exists and add chunk information
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            doc.metadata['chunk'] = i
            doc.metadata['original_length'] = len(original_content)
            doc.metadata['cleaned_length'] = len(cleaned_content)
            
            processed_docs.append(doc)
        
        logger.info(f"Preprocessing completed: {len(processed_docs)} documents, {skipped_count} skipped")
        return processed_docs


class ReRanker:
    """Cross-encoder re-ranking utilities with fallback options"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.cross_encoder = None
        self.sentence_transformer = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading cross-encoder model: {model_name}")
                self.cross_encoder = CrossEncoder(model_name)
                logger.info("Cross-encoder model loaded successfully")
                
                # Initialize sentence transformer for similarity scoring
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self.cross_encoder = None
                self.sentence_transformer = None
        else:
            logger.warning("sentence-transformers not available. Re-ranking will use fallback methods.")
    
    def rerank_documents(self, query: str, candidate_docs: List[Document], top_k: int = None) -> List[Document]:
        """Re-rank documents using cross-encoder or fallback method"""
        if not candidate_docs:
            return candidate_docs
        
        if top_k is None:
            top_k = len(candidate_docs)
        
        if self.cross_encoder:
            return self._cross_encoder_rerank(query, candidate_docs)[:top_k]
        else:
            return self._fallback_rerank(query, candidate_docs)[:top_k]
    
    def _cross_encoder_rerank(self, query: str, candidate_docs: List[Document]) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in candidate_docs:
                # Truncate document content if too long (cross-encoder has token limits)
                doc_text = doc.page_content[:512]  # Limit to 512 characters
                query_doc_pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Create list of (document, score) pairs
            doc_score_pairs = list(zip(candidate_docs, cross_encoder_scores))
            
            # Sort by cross-encoder score (descending)
            doc_score_pairs.sort(key=lambda x: float(x[1]), reverse=True)
            
            # Return sorted documents
            reranked_docs = [doc for doc, score in doc_score_pairs]
            
            logger.debug(f"Cross-encoder re-ranking completed. Top scores: {cross_encoder_scores[:3]}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during cross-encoder re-ranking: {e}")
            return candidate_docs
    
    def _fallback_rerank(self, query: str, candidate_docs: List[Document]) -> List[Document]:
        """Fallback re-ranking using simple text similarity"""
        try:
            if self.sentence_transformer:
                # Use sentence transformer if available
                query_emb = self.sentence_transformer.encode([query])
                doc_embs = self.sentence_transformer.encode([doc.page_content[:512] for doc in candidate_docs])
                
                similarities = cosine_similarity(query_emb, doc_embs)[0]
                
                # Create list of (document, similarity) pairs
                doc_sim_pairs = list(zip(candidate_docs, similarities))
                
                # Sort by similarity (descending)
                doc_sim_pairs.sort(key=lambda x: float(x[1]), reverse=True)
                
                return [doc for doc, sim in doc_sim_pairs]
            else:
                # Simple lexical overlap as last resort
                return self._lexical_overlap_rerank(query, candidate_docs)
                
        except Exception as e:
            logger.error(f"Error during fallback re-ranking: {e}")
            return candidate_docs
    
    def _lexical_overlap_rerank(self, query: str, candidate_docs: List[Document]) -> List[Document]:
        """Simple lexical overlap re-ranking"""
        query_words = set(query.lower().split())
        
        doc_scores = []
        for doc in candidate_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            doc_scores.append((doc, overlap))
        
        # Sort by overlap (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores]
    
    def calculate_similarity_score(self, answer: str, ground_truth: str) -> float:
        """Calculate cosine similarity between answer and ground truth"""
        if not answer or not ground_truth:
            return 0.0
        
        try:
            if self.sentence_transformer:
                answer_emb = self.sentence_transformer.encode([answer])
                gt_emb = self.sentence_transformer.encode([ground_truth])
                similarity = cosine_similarity(answer_emb, gt_emb)[0][0]
                return max(0.0, float(similarity))
            else:
                # Fallback to simple word overlap
                answer_words = set(answer.lower().split())
                gt_words = set(ground_truth.lower().split())
                
                if not answer_words or not gt_words:
                    return 0.0
                
                overlap = len(answer_words.intersection(gt_words))
                union = len(answer_words.union(gt_words))
                
                return overlap / union if union > 0 else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


def reciprocal_rank_fusion(dense_results: List[Tuple[int, float]], 
                          sparse_results: List[Tuple[int, float]], 
                          k: int = 60) -> List[Tuple[int, float]]:
    """Implement Reciprocal Rank Fusion with proper handling"""
    
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    
    # Process dense results
    for rank, (doc_idx, score) in enumerate(dense_results):
        rrf_scores[doc_idx] += 1.0 / (rank + k)
    
    # Process sparse results
    for rank, (doc_idx, score) in enumerate(sparse_results):
        rrf_scores[doc_idx] += 1.0 / (rank + k)
    
    # Sort by RRF score (descending)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


# Utility functions for text preprocessing
def preprocess_query(query: str) -> str:
    """Preprocess query for better matching"""
    if not query:
        return ""
    
    query = str(query).strip()
    
    # Remove common question words
    query = re.sub(r'\b(what|how|why|when|where|who)\b', '', query, flags=re.IGNORECASE)
    
    # Remove question marks
    query = query.replace("?", "").strip()
    
    # Clean whitespace
    query = re.sub(r'\s+', ' ', query)
    
    return query


# Example usage and testing functions
def test_bm25():
    """Test BM25 functionality"""
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outran a quick fox",
        "The dog was lazy but the fox was quick",
        "Programming with Python is fun and easy"
    ]
    
    bm25 = BM25Retriever()
    bm25.fit(documents)
    
    query = "quick fox"
    scores = bm25.get_scores(query)
    top_docs = bm25.get_top_k(query, k=2)
    
    print(f"Query: {query}")
    print(f"Scores: {scores}")
    print(f"Top documents: {top_docs}")


if __name__ == "__main__":
    # Run test
    test_bm25()