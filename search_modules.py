import logging
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod
from langchain.schema import Document

# Milvus imports
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Import BM25 and cross-encoder from preprocessor
from preprocessor import BM25Retriever

logger = logging.getLogger(__name__)


class BaseSearchModule(ABC):
    """Abstract base class for search modules"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add documents to the search index"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        pass


class MilvusBaseModule(BaseSearchModule, ABC):
    """Base class for Milvus-based search modules"""
    
    def __init__(self, collection_name: str, embeddings, host: str = "localhost", port: str = "19530"):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embeddings = embeddings
        self.collection = None
        self.embedding_dim = 1024  # Cohere embed-english-v3.0 dimension
        self.documents = []  # Store documents for retrieval
        
        # Connect to Milvus
        self._connect()
        self._load_or_create_collection()
        
    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection(self):
        """Create collection with schema"""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        # Create schema
        schema = CollectionSchema(fields, "Document embeddings collection")
        
        # Create collection
        self.collection = Collection(self.collection_name, schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index("embedding", index_params)
        logger.info(f"Created collection: {self.collection_name}")
    
    def _load_or_create_collection(self):
        """Load existing collection or create new one"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            self._create_collection()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to Milvus collection: {self.collection_name}")
        
        # Store documents for later retrieval
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # Process in batches for vector store
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            # Get embeddings for batch
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            
            if batch_embeddings:
                # Prepare data for insertion
                data = [
                    list(range(start_idx + i, start_idx + i + len(batch_docs))),  # doc_id
                    [doc.page_content for doc in batch_docs],  # content
                    [json.dumps(doc.metadata) for doc in batch_docs],  # metadata as JSON string
                    batch_embeddings  # embedding vectors
                ]
                
                # Insert data
                self.collection.insert(data)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        # Flush to ensure data is written
        self.collection.flush()
        logger.info(f"Successfully added {len(documents)} documents to Milvus")

    def _dense_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """Perform dense vector search using embeddings"""
        try:
            # Load collection into memory for search
            self.collection.load()
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query: {query}")
                return []
            
            # Search parameters for dense search
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            }
            
            # Perform dense search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["doc_id", "content", "metadata"]
            )
            
            # Process dense results
            dense_results = []
            for hits in results:
                for hit in hits:
                    doc_id = int(hit.entity.get("doc_id"))
                    similarity = float(hit.score)
                    dense_results.append((doc_id, similarity))
            
            return dense_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class HybridSearchModule(MilvusBaseModule):
    """Hybrid search with dense + sparse search and cross-encoder re-ranking"""
    
    def __init__(self, collection_name: str, embeddings, host: str = "localhost", port: str = "19530"):
        super().__init__(collection_name, embeddings, host, port)
        
        # Initialize BM25 for sparse retrieval
        self.bm25_retriever = BM25Retriever()
        self.is_bm25_fitted = False
        
        # Initialize cross-encoder for re-ranking
        self._initialize_cross_encoder()
        
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder for re-ranking"""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder initialized successfully")
        except ImportError:
            logger.warning("sentence-transformers not available. Cross-encoder re-ranking disabled.")
            self.cross_encoder = None
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.cross_encoder = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both vector store and BM25 index"""
        super().add_documents(documents)
        
        # Extract text content for BM25
        document_texts = [doc.page_content for doc in self.documents]
        
        # Fit BM25 on all documents
        logger.info("Fitting BM25 on document corpus...")
        self.bm25_retriever.fit(document_texts)
        self.is_bm25_fitted = True
        logger.info("BM25 fitting completed")
    
    def search(self, query: str, top_k: int = 5, rerank_top_k: int = 20) -> List[Dict]:
        """Retrieve documents using hybrid search with RRF and cross-encoder re-ranking"""
        # Get results from both dense and sparse search
        dense_results = self._dense_search(query, rerank_top_k)
        sparse_results = self._sparse_search(query, rerank_top_k)
        
        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Get candidate documents for re-ranking
        candidate_docs = self._get_candidate_documents(fused_results[:rerank_top_k])
        
        # Apply cross-encoder re-ranking if available
        if self.cross_encoder and candidate_docs:
            reranked_docs = self._cross_encoder_rerank(query, candidate_docs)
        else:
            # Fallback to RRF scores if no cross-encoder
            reranked_docs = candidate_docs
        
        # Return top_k documents
        return reranked_docs[:top_k]
    
    def _sparse_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """Perform sparse search using BM25"""
        try:
            if not self.is_bm25_fitted:
                logger.warning("BM25 not fitted. Skipping sparse search.")
                return []
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_retriever.get_scores(query)
            
            # Create (doc_id, score) pairs and sort by score
            scored_docs = [(i, score) for i, score in enumerate(bm25_scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            return scored_docs[:limit]
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]], 
                              sparse_results: List[Tuple[int, float]], 
                              k: int = 60) -> List[Tuple[int, float]]:
        """Apply Reciprocal Rank Fusion to combine dense and sparse results"""
        rrf_scores = defaultdict(float)
        
        # Process dense results
        for rank, (doc_id, score) in enumerate(dense_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Process sparse results
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score descending
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _get_candidate_documents(self, fused_results: List[Tuple[int, float]]) -> List[Dict]:
        """Retrieve full document information for candidate documents"""
        candidate_docs = []
        
        for doc_id, rrf_score in fused_results:
            try:
                # Get document from stored documents list
                if 0 <= doc_id < len(self.documents):
                    doc = self.documents[doc_id]
                    candidate = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": rrf_score,  # Use RRF score as initial similarity
                        "doc_id": doc_id
                    }
                    candidate_docs.append(candidate)
                else:
                    logger.warning(f"Doc ID {doc_id} out of range")
                    
            except Exception as e:
                logger.error(f"Failed to retrieve document {doc_id}: {e}")
                continue
        
        return candidate_docs
    
    def _cross_encoder_rerank(self, query: str, candidate_docs: List[Dict]) -> List[Dict]:
        """Re-rank candidate documents using cross-encoder"""
        if not candidate_docs:
            return []
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in candidate_docs:
                # Truncate document content if too long
                doc_text = doc["text"][:512]  # Limit to 512 characters
                query_doc_pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            rerank_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Update similarity scores and sort
            for i, doc in enumerate(candidate_docs):
                doc["similarity"] = float(rerank_scores[i])
            
            # Sort by cross-encoder score descending
            candidate_docs.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.debug(f"Cross-encoder re-ranking completed. Top scores: {[doc['similarity'] for doc in candidate_docs[:3]]}")
            return candidate_docs
            
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            # Fallback: return documents sorted by RRF scores
            return sorted(candidate_docs, key=lambda x: x["similarity"], reverse=True)


class HybridSearchNoRerankModule(MilvusBaseModule):
    """Hybrid search with dense + sparse search but without cross-encoder re-ranking"""
    
    def __init__(self, collection_name: str, embeddings, host: str = "localhost", port: str = "19530"):
        super().__init__(collection_name, embeddings, host, port)
        
        # Initialize BM25 for sparse retrieval
        self.bm25_retriever = BM25Retriever()
        self.is_bm25_fitted = False
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both vector store and BM25 index"""
        super().add_documents(documents)
        
        # Extract text content for BM25
        document_texts = [doc.page_content for doc in self.documents]
        
        # Fit BM25 on all documents
        logger.info("Fitting BM25 on document corpus...")
        self.bm25_retriever.fit(document_texts)
        self.is_bm25_fitted = True
        logger.info("BM25 fitting completed")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using hybrid search with RRF but no re-ranking"""
        # Get results from both dense and sparse search
        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)
        
        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Get candidate documents
        candidate_docs = self._get_candidate_documents(fused_results[:top_k])
        
        return candidate_docs
    
    def _sparse_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """Perform sparse search using BM25"""
        try:
            if not self.is_bm25_fitted:
                logger.warning("BM25 not fitted. Skipping sparse search.")
                return []
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_retriever.get_scores(query)
            
            # Create (doc_id, score) pairs and sort by score
            scored_docs = [(i, score) for i, score in enumerate(bm25_scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            return scored_docs[:limit]
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]], 
                              sparse_results: List[Tuple[int, float]], 
                              k: int = 60) -> List[Tuple[int, float]]:
        """Apply Reciprocal Rank Fusion to combine dense and sparse results"""
        rrf_scores = defaultdict(float)
        
        # Process dense results
        for rank, (doc_id, score) in enumerate(dense_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Process sparse results
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score descending
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _get_candidate_documents(self, fused_results: List[Tuple[int, float]]) -> List[Dict]:
        """Retrieve full document information for candidate documents"""
        candidate_docs = []
        
        for doc_id, rrf_score in fused_results:
            try:
                # Get document from stored documents list
                if 0 <= doc_id < len(self.documents):
                    doc = self.documents[doc_id]
                    candidate = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": rrf_score,  # Use RRF score as similarity
                        "doc_id": doc_id
                    }
                    candidate_docs.append(candidate)
                else:
                    logger.warning(f"Doc ID {doc_id} out of range")
                    
            except Exception as e:
                logger.error(f"Failed to retrieve document {doc_id}: {e}")
                continue
        
        return candidate_docs


class SemanticSearchModule(MilvusBaseModule):
    """Pure semantic search using only dense vector search"""
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using only semantic search"""
        # Get results from dense search only
        dense_results = self._dense_search(query, top_k)
        
        # Convert to document format
        candidate_docs = []
        
        for doc_id, similarity in dense_results:
            try:
                # Get document from stored documents list
                if 0 <= doc_id < len(self.documents):
                    doc = self.documents[doc_id]
                    candidate = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": similarity,
                        "doc_id": doc_id
                    }
                    candidate_docs.append(candidate)
                else:
                    logger.warning(f"Doc ID {doc_id} out of range")
                    
            except Exception as e:
                logger.error(f"Failed to retrieve document {doc_id}: {e}")
                continue
        
        return candidate_docs


class KeywordSearchModule(BaseSearchModule):
    """Pure keyword search using only BM25"""
    
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.is_fitted = False
        self.documents = []
    
    def add_documents(self, documents: List[Document]):
        """This method is called by the main pipeline but BM25 fitting happens in fit_documents"""
        self.documents = documents
    
    def fit_documents(self, document_texts: List[str]):
        """Fit BM25 on document texts"""
        logger.info("Fitting BM25 for keyword search...")
        self.bm25_retriever.fit(document_texts)
        self.is_fitted = True
        logger.info("BM25 fitting completed for keyword search")
    
    def search(self, query: str, documents: List[Document], top_k: int = 5) -> List[Dict]:
        """Retrieve documents using only BM25 keyword search"""
        if not self.is_fitted:
            logger.error("BM25 not fitted. Cannot perform keyword search.")
            return []
        
        try:
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_retriever.get_scores(query)
            
            # Create (doc_id, score) pairs and sort by score
            scored_docs = [(i, score) for i, score in enumerate(bm25_scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Convert top results to document format
            candidate_docs = []
            
            for doc_id, bm25_score in scored_docs[:top_k]:
                try:
                    if 0 <= doc_id < len(documents):
                        doc = documents[doc_id]
                        candidate = {
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity": bm25_score,  # Use BM25 score as similarity
                            "doc_id": doc_id
                        }
                        candidate_docs.append(candidate)
                    else:
                        logger.warning(f"Doc ID {doc_id} out of range")
                        
                except Exception as e:
                    logger.error(f"Failed to retrieve document {doc_id}: {e}")
                    continue
            
            return candidate_docs
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []