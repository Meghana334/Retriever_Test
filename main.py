# import os
# import json
# import pandas as pd
# from datetime import datetime
# from tqdm import tqdm
# from dotenv import load_dotenv
# import logging

# from langchain.schema import Document
# from milvus import CohereMilvusRetriever
# from preprocessor import CohereEmbeddings, chunk_pdf
# from embed_utils import calculate_text_similarity

# # Logger setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# load_dotenv()

# # Load models
# cohere_embedder = CohereEmbeddings()
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# if not COHERE_API_KEY:
#     raise ValueError("COHERE_API_KEY environment variable is not set.")

# DEFAULT_COLLECTION_NAME = "test_collection"
# vectorstore = CohereMilvusRetriever(collection_name=DEFAULT_COLLECTION_NAME, embeddings=cohere_embedder)


# def to_vector_store(pdf_path=None):
#     logger.info("▶ Starting hybrid RAG pipeline (Milvus + Cohere)...")

#     documents = []
#     if pdf_path and os.path.exists(pdf_path):
#         logger.info("📄 Chunking PDF...")
#         documents = chunk_pdf(pdf_path)

#     if not documents:
#         logger.error("❌ No valid documents loaded. Aborting pipeline.")
#         return

#     logger.info("💾 Inserting into Milvus...")
#     docs = [
#         Document(
#             page_content=chunk.page_content,
#             metadata={
#                 'page': chunk.metadata['page'],
#                 'chunk': chunk.metadata['chunk'],
#             }
#         )
#         for chunk in documents
#     ]
#     vectorstore.add_documents(docs)


# def run_pipeline(pdf_path=None, csv_path=None, output_csv=None, output_json=None):
#     to_vector_store(pdf_path)

#     df = pd.read_csv(csv_path)
#     all_metadata = []

#     logger.info("🔍 Processing queries...")
#     retrieved_answers_k_1 = []
#     retrieved_answers_k_2 = []
#     retrieved_answers_k_3 = []
#     similarity_scores_k_1_list = []
#     similarity_scores_k_2_list = []
#     similarity_scores_k_3_list = []
#     distances_k_1_list = []
#     distances_k_2_list = []
#     distances_k_3_list = []

#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         question = str(row['QUESTION']).strip()
#         ground_truth = str(row['GROUND TRUTH']).strip()
#         if not question or question.lower() == 'nan':
#             logger.warning(f"Skipping empty question at index {idx}")
#             continue

#         results = vectorstore.retrieve(question, top_k=3)
#         if not results or len(results) < 3:
#             logger.warning(f"Less than 3 results found for question at index {idx}")
#             continue

#         answers_meta = []

#         for k in range(3):
#             result = results[k]
#             answer = result.get("text", "").strip()
#             meta = result.get("metadata", {})
#             page = meta.get("page", "N/A")
#             chunk = meta.get("chunk", "N/A")
#             sim_score, dist = calculate_text_similarity(answer, ground_truth)

#             answers_meta.append({
#                 "answer": answer,
#                 "page": page,
#                 "chunk_no": chunk,
#                 "similarity_score": sim_score.item(),
#                 "distance": dist.item()
#             })

#             # Collect for CSV export
#             if k == 0:
#                 retrieved_answers_k_1.append(answer)
#                 similarity_scores_k_1_list.append(sim_score.item())
#                 distances_k_1_list.append(dist.item())
#             elif k == 1:
#                 retrieved_answers_k_2.append(answer)
#                 similarity_scores_k_2_list.append(sim_score.item())
#                 distances_k_2_list.append(dist.item())
#             elif k == 2:
#                 retrieved_answers_k_3.append(answer)
#                 similarity_scores_k_3_list.append(sim_score.item())
#                 distances_k_3_list.append(dist.item())

#         all_metadata.append({
#             "question": question,
#             "ground_truth": ground_truth,
#             "retrieved_answers": answers_meta
#         })

#     # Create CSV
#     results_df = pd.DataFrame({
#         'QUESTION': df['QUESTION'],
#         'GROUND TRUTH': df['GROUND TRUTH'],
#         'RETRIEVED ANSWER K=1': retrieved_answers_k_1,
#         'RETRIEVED ANSWER K=2': retrieved_answers_k_2,
#         'RETRIEVED ANSWER K=3': retrieved_answers_k_3,
#         'SIMILARITY SCORE K=1': similarity_scores_k_1_list,
#         'SIMILARITY SCORE K=2': similarity_scores_k_2_list,
#         'SIMILARITY SCORE K=3': similarity_scores_k_3_list,
#         'DISTANCE K=1': distances_k_1_list,
#         'DISTANCE K=2': distances_k_2_list,
#         'DISTANCE K=3': distances_k_3_list,
#     })

#     # Save to CSV
#     results_df.to_csv(output_csv, index=False)
#     logger.info(f"📁 CSV results saved to {output_csv}")

#     # Save to JSON
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(all_metadata, f, ensure_ascii=False, indent=4)
#     logger.info(f"📁 Metadata JSON saved to {output_json}")


# if __name__ == "__main__":
#     pdf_path = "/mnt/c/Users/sanja/Downloads/BL9000-Owners-Manual.pdf"
#     csv_path = r"/mnt/c/Users/sanja/Downloads/hybrid search DATASET2 - DATASET.csv"

#     logger.info("🚀 Running pipeline...")
#     run_pipeline(
#         pdf_path=pdf_path,
#         csv_path=csv_path,
#         output_csv="hybrid_similarity_comparison.csv",
#         output_json="hybrid_retrieved_metadata.json"
#     )


import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional

from langchain.schema import Document
from milvus_utils import CohereMilvusRetriever
from preprocessor import CohereEmbeddings, DocumentProcessor, ReRanker
from embed_utils import calculate_text_similarity

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set.")

DEFAULT_COLLECTION_NAME = "hybrid_documents"


class HybridRAGPipeline:
    """Hybrid RAG Pipeline with Milvus Vector Store and Cohere Embeddings"""
    
    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME, 
                 milvus_host: str = "localhost", milvus_port: str = "19530"):
        self.collection_name = collection_name
        self.cohere_embedder = CohereEmbeddings(COHERE_API_KEY)
        self.vector_store = CohereMilvusRetriever(
            collection_name=collection_name, 
            embeddings=self.cohere_embedder,
            host=milvus_host, 
            port=milvus_port
        )
        self.doc_processor = DocumentProcessor()
        self.reranker = ReRanker()
        self.documents = []
        logger.info(f"✅ Initialized Hybrid RAG Pipeline with collection: {collection_name}")
    
    def load_documents_to_vector_store(self, pdf_path: str):
        """Load PDF documents into Milvus vector store"""
        logger.info("▶ Starting hybrid RAG pipeline (Milvus + Cohere)...")
        
        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"❌ PDF file not found: {pdf_path}")
            return
        
        logger.info("📄 Chunking PDF...")
        documents = self.doc_processor.load_pdf(pdf_path)
        self.documents = self.doc_processor.preprocess_documents(documents)
        
        if not self.documents:
            logger.error("❌ No valid documents loaded. Aborting pipeline.")
            return
        
        logger.info("💾 Inserting documents into Milvus...")
        # Add documents to vector store (it handles batching internally)
        self.vector_store.add_documents(self.documents)
        
        logger.info(f"✅ Successfully loaded {len(self.documents)} documents to vector store")
    
    def retrieve_documents(self, query: str, top_k: int = 3, rerank_top_k: int = 15) -> List[Dict]:
        """Retrieve top-k documents using hybrid search"""
        return self.vector_store.retrieve(query, top_k=top_k, rerank_top_k=rerank_top_k)
    
    def calculate_similarity_score(self, answer: str, ground_truth: str) -> tuple:
        """Calculate similarity score and distance between answer and ground truth"""
        similarity, distance = calculate_text_similarity(answer, ground_truth)
        return similarity, distance


def run_pipeline(pdf_path: str = None, csv_path: str = None, 
                output_csv: str = None, output_json: str = None):
    """Run the complete hybrid RAG pipeline"""
    
    # Initialize pipeline
    pipeline = HybridRAGPipeline()
    
    # Load documents to vector store
    if pdf_path:
        pipeline.load_documents_to_vector_store(pdf_path)
    else:
        logger.warning("No PDF path provided, skipping document loading")
    
    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read CSV file
    logger.info(f"📊 Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"📈 Loaded {len(df)} questions from CSV")
    
    # Initialize result lists
    all_metadata = []
    retrieved_answers_k_1 = []
    retrieved_answers_k_2 = []
    retrieved_answers_k_3 = []
    similarity_scores_k_1_list = []
    similarity_scores_k_2_list = []
    similarity_scores_k_3_list = []
    distances_k_1_list = []
    distances_k_2_list = []
    distances_k_3_list = []
    
    logger.info("🔍 Processing queries with hybrid search...")
    
    # Process each question
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions with hybrid retrieval"):
        question = str(row['GROUND TRUTH']).strip()
        ground_truth = str(row['GROUND TRUTH']).strip()
        
        # Skip invalid questions
        if not question or question.lower() == 'nan':
            logger.warning(f"⚠️ Skipping empty question at index {idx}")
            # Append empty values to maintain dataframe alignment
            retrieved_answers_k_1.append("")
            retrieved_answers_k_2.append("")
            retrieved_answers_k_3.append("")
            similarity_scores_k_1_list.append(0.0)
            similarity_scores_k_2_list.append(0.0)
            similarity_scores_k_3_list.append(0.0)
            distances_k_1_list.append(1.0)
            distances_k_2_list.append(1.0)
            distances_k_3_list.append(1.0)
            continue
        
        # Retrieve documents using hybrid search
        results = pipeline.retrieve_documents(question, top_k=3, rerank_top_k=15)
        
        # Handle insufficient results
        while len(results) < 3:
            results.append({
                "text": "No answer found",
                "metadata": {"page": "N/A", "chunk": "N/A"},
                "similarity": 0.0,
                "doc_id": -1
            })
        
        # Process results for each k value
        answers_meta = []
        
        for k in range(3):
            result = results[k]
            answer = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            page = metadata.get("page", "N/A")
            chunk = metadata.get("chunk", "N/A")
            
            # Calculate similarity scores
            sim_score, distance = pipeline.calculate_similarity_score(answer, ground_truth)
            
            # Store metadata
            answers_meta.append({
                "answer": answer,
                "page": page,
                "chunk_no": chunk,
                "similarity_score": float(sim_score),
                "distance": float(distance),
                "hybrid_relevance_score": result.get("similarity", 0.0)  # Add hybrid search relevance
            })
            
            # Collect for CSV export based on k value
            if k == 0:
                retrieved_answers_k_1.append(answer)
                similarity_scores_k_1_list.append(float(sim_score))
                distances_k_1_list.append(float(distance))
            elif k == 1:
                retrieved_answers_k_2.append(answer)
                similarity_scores_k_2_list.append(float(sim_score))
                distances_k_2_list.append(float(distance))
            elif k == 2:
                retrieved_answers_k_3.append(answer)
                similarity_scores_k_3_list.append(float(sim_score))
                distances_k_3_list.append(float(distance))
        
        # Store complete metadata for JSON export
        all_metadata.append({
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_answers": answers_meta
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'QUESTION': df['QUESTION'],
        'GROUND TRUTH': df['GROUND TRUTH'],
        'RETRIEVED ANSWER K=1': retrieved_answers_k_1,
        'RETRIEVED ANSWER K=2': retrieved_answers_k_2,
        'RETRIEVED ANSWER K=3': retrieved_answers_k_3,
        'SIMILARITY SCORE K=1': similarity_scores_k_1_list,
        'SIMILARITY SCORE K=2': similarity_scores_k_2_list,
        'SIMILARITY SCORE K=3': similarity_scores_k_3_list,
        'DISTANCE K=1': distances_k_1_list,
        'DISTANCE K=2': distances_k_2_list,
        'DISTANCE K=3': distances_k_3_list,
    })
    
    # Save results to CSV
    if not output_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = f"hybrid_similarity_results_{timestamp}.csv"
    
    results_df.to_csv(output_csv, index=False)
    logger.info(f"📁 CSV results saved to {output_csv}")
    
    # Save metadata to JSON
    if not output_json:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_json = f"hybrid_retrieved_metadata_{timestamp}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    logger.info(f"📁 Metadata JSON saved to {output_json}")
    
    # Print summary statistics
    logger.info("📊 HYBRID SEARCH SUMMARY STATISTICS:")
    logger.info(f"Total questions processed: {len(results_df)}")
    
    for k in [1, 2, 3]:
        scores = results_df[f'SIMILARITY SCORE K={k}']
        valid_scores = scores[scores > 0]
        if len(valid_scores) > 0:
            logger.info(f"K={k} - Avg similarity: {valid_scores.mean():.4f}, "
                       f"Max: {valid_scores.max():.4f}, "
                       f"Min: {valid_scores.min():.4f}")
    
    # Additional hybrid search statistics
    hybrid_scores = [item['retrieved_answers'] for item in all_metadata]
    if hybrid_scores:
        k1_hybrid_scores = [item[0]['hybrid_relevance_score'] for item in hybrid_scores if item]
        if k1_hybrid_scores:
            avg_hybrid_score = sum(k1_hybrid_scores) / len(k1_hybrid_scores)
            logger.info(f"Average hybrid relevance score (K=1): {avg_hybrid_score:.4f}")
    
    return results_df, output_csv, output_json


def main():
    """Main execution function"""
    print("🚀 Starting Hybrid RAG Pipeline with Milvus + Cohere + BM25...")
    
    # Configuration
    PDF_PATH = "/mnt/c/Users/sanja/Downloads/BL9000-Owners-Manual.pdf"  # Update with your PDF path
    CSV_PATH = r"/mnt/c/Users/sanja/Downloads/topicwise-csv - DATASET1 (1).csv"  # Update with your CSV path  
    
    try:
        # Check if CSV exists and run pipeline
        if CSV_PATH and os.path.exists(CSV_PATH):
            logger.info("📊 Running hybrid search pipeline with CSV processing...")
            results_df, output_csv, output_json = run_pipeline(
                pdf_path=PDF_PATH,
                csv_path=CSV_PATH,
                output_csv="topicwise_hybrid_similarity_comparison.csv",
                output_json="topicwise_hybrid_retrieved_metadata.json"
            )
            
            # Print completion message
            logger.info("🎉 Hybrid RAG pipeline completed successfully!")
            logger.info(f"📈 Results saved to: {output_csv}")
            logger.info(f"📋 Metadata saved to: {output_json}")
            
        else:
            logger.error(f"❌ CSV file not found: {CSV_PATH}")
            logger.info("Please update the CSV_PATH variable with the correct path to your dataset.")
    
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()