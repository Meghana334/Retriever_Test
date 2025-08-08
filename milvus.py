# # milvus.py

# from typing import Any, Dict, List
# from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
# import json
# import logging
# import os

# from langchain.vectorstores import Milvus
# from langchain.embeddings.cohere import CohereEmbeddings
# from langchain.schema import Document
# from langchain.schema import Document
# from langchain.embeddings.base import Embeddings
# from langchain_milvus import Milvus
# from pymilvus import connections, exceptions as milvus_exceptions

# from dotenv import load_dotenv
# load_dotenv()

# COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
# MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# URI = "http://localhost:19530"


# def safe_connect(alias: str = "default", host: str = MILVUS_HOST, port: str = MILVUS_PORT):
#     """
#     Connect to Milvus under `alias`. If alias exists with different config,
#     disconnect and reconnect with new parameters.
#     """
#     try:
#         connections.connect(alias=alias, host=host, port=port)
#     except milvus_exceptions.ConnectionConfigException as e:
#         msg = str(e)
#         if "Alias of" in msg and "not the same as passed in" in msg:
#             connections.disconnect(alias)
#             connections.connect(alias=alias, host=host, port=port)
#         else:
#             raise
        
# class CohereMilvusRetriever:
#     """
#     Retriever that wraps Cohere embeddings with LangChain's Milvus vector store.
#     """
#     def __init__(
#         self,
#         collection_name: str,
#         alias: str = "default",
#         host: str = MILVUS_HOST,
#         port: str = MILVUS_PORT,
#         top_k: int = 4,
#         embeddings: Embeddings = None,
#     ):
#         # Safely connect to Milvus
#         safe_connect(alias=alias, host=host, port=port)
#         logger.info(f"Connected to Milvus at {host}:{port} with alias '{alias}'")
#         # Adapter and Vector store
#         self.vector_store = Milvus(
#             embedding_function=embeddings,
#             connection_args={"uri": URI},
#             collection_name=collection_name,
#             # drop_old_collection=False,
#             index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
#             search_params={"metric_type": "L2", "params": {"nprobe": 10}},
#             auto_id=True,
#             drop_old=True, #Todo uncomment this
#             vector_field="embedding", # Ensure this matches your embedding field name
#             text_field="text", #Error still message=Attempt to insert an unexpected field `text` 
#             metadata_field="metadata",
#         )
#         # LangChain retriever
#         self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

#     def add_documents(self, docs: List[Document]) -> None:
#         """
#         Adds documents to Milvus. Document.metadata must be JSON-serializable.
#         """
#         for doc in docs:
#             sanitized = {}
#             for k, v in (doc.metadata or {}).items():
#                 if isinstance(v, (list, dict)):
#                     sanitized[k] = json.dumps(v)
#                 else:
#                     sanitized[k] = v
#             doc.metadata = sanitized
#         self.vector_store.add_documents(docs, nullable=True)

#     def get_relevant_documents(self, query: str) -> List[Document]:
#         """
#         Returns LangChain Documents with metadata.
#         """
#         return self.retriever.get_relevant_documents(query)

#     def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
#         """
#         Convenience: returns list of dicts with 'page', 'text', 'image_path'.
#         """
#         docs = self.get_relevant_documents(query)
#         print(docs)
#         results = []
#         for doc in docs[: top_k or len(docs)]:
#             results.append({
#                 # "page": doc.metadata.get("page"), #TODO: Handle page numbers if needed
#                 "text": doc.page_content,
#                 # "image_path": json.loads(doc.metadata.get("image_path", "[]")),
#                 # "reference_path": doc.metadata.get("reference_path")
#             })
#         return results



import logging
import json
from typing import List, Dict, Tuple
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

# Fix for LangChain deprecation warnings
try:
    from langchain_community.vectorstores import Milvus
    from langchain_community.embeddings import CohereEmbeddings as LangChainCohereEmbeddings
except ImportError:
    from langchain.vectorstores import Milvus
    from langchain.embeddings.cohere import CohereEmbeddings as LangChainCohereEmbeddings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus vector store for persistent document storage and retrieval"""
    
    def __init__(self, collection_name: str = "document_embeddings", host: str = "localhost", port: str = "19530"):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        self.embedding_dim = 1024  # Cohere embed-english-v3.0 dimension
        
        # Connect to Milvus
        self._connect()
        
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
    
    def insert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Insert documents and their embeddings into Milvus"""
        if not self.collection:
            self._load_or_create_collection()
        
        # Prepare data for insertion
        data = [
            list(range(len(documents))),  # doc_id
            [doc.page_content for doc in documents],  # content
            [json.dumps(doc.metadata) for doc in documents],  # metadata as JSON string
            embeddings  # embedding vectors
        ]
        
        # Insert data
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Inserted {len(documents)} documents into Milvus")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, float, str, dict]]:
        """Search for similar documents"""
        if not self.collection:
            self._load_or_create_collection()
        
        # Load collection into memory for search
        self.collection.load()
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "content", "metadata"]
        )
        
        # Process results
        search_results = []
        for hits in results:
            for hit in hits:
                doc_id = hit.entity.get("doc_id")
                content = hit.entity.get("content")
                metadata = json.loads(hit.entity.get("metadata"))
                similarity = hit.score
                
                search_results.append((doc_id, similarity, content, metadata))
        
        return search_results
    
    def delete_collection(self):
        """Delete the collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_stats(self):
        """Get collection statistics"""
        if not self.collection:
            self._load_or_create_collection()
        
        stats = self.collection.num_entities
        return {"num_documents": stats}