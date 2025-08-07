# milvus.py

from typing import Any, Dict, List
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import json
import logging
import os

from langchain.vectorstores import Milvus
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.schema import Document
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_milvus import Milvus
from pymilvus import connections, exceptions as milvus_exceptions

from dotenv import load_dotenv
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
URI = "http://localhost:19530"


def safe_connect(alias: str = "default", host: str = MILVUS_HOST, port: str = MILVUS_PORT):
    """
    Connect to Milvus under `alias`. If alias exists with different config,
    disconnect and reconnect with new parameters.
    """
    try:
        connections.connect(alias=alias, host=host, port=port)
    except milvus_exceptions.ConnectionConfigException as e:
        msg = str(e)
        if "Alias of" in msg and "not the same as passed in" in msg:
            connections.disconnect(alias)
            connections.connect(alias=alias, host=host, port=port)
        else:
            raise
        
class CohereMilvusRetriever:
    """
    Retriever that wraps Cohere embeddings with LangChain's Milvus vector store.
    """
    def __init__(
        self,
        collection_name: str,
        alias: str = "default",
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        top_k: int = 4,
        embeddings: Embeddings = None,
    ):
        # Safely connect to Milvus
        safe_connect(alias=alias, host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port} with alias '{alias}'")
        # Adapter and Vector store
        self.vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI},
            collection_name=collection_name,
            # drop_old_collection=False,
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
            search_params={"metric_type": "L2", "params": {"nprobe": 10}},
            auto_id=True,
            drop_old=True, #Todo uncomment this
            vector_field="embedding", # Ensure this matches your embedding field name
            text_field="text", #Error still message=Attempt to insert an unexpected field `text` 
            metadata_field="metadata",
        )
        # LangChain retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def add_documents(self, docs: List[Document]) -> None:
        """
        Adds documents to Milvus. Document.metadata must be JSON-serializable.
        """
        for doc in docs:
            sanitized = {}
            for k, v in (doc.metadata or {}).items():
                if isinstance(v, (list, dict)):
                    sanitized[k] = json.dumps(v)
                else:
                    sanitized[k] = v
            doc.metadata = sanitized
        self.vector_store.add_documents(docs, nullable=True)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Returns LangChain Documents with metadata.
        """
        return self.retriever.get_relevant_documents(query)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Convenience: returns list of dicts with 'page', 'text', 'image_path'.
        """
        docs = self.get_relevant_documents(query)
        print(docs)
        results = []
        for doc in docs[: top_k or len(docs)]:
            results.append({
                # "page": doc.metadata.get("page"), #TODO: Handle page numbers if needed
                "text": doc.page_content,
                # "image_path": json.loads(doc.metadata.get("image_path", "[]")),
                # "reference_path": doc.metadata.get("reference_path")
            })
        return results