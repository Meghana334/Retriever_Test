import os
# from dotenv import load_dotenv

import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer #TODO improve Speed 
import logging
# import cohere
from langchain.schema import Document
from milvus import CohereMilvusRetriever
from preprocessor import CohereEmbeddings, chunk_pdf
# from milvus_utils import list_collections
from embed_utils import calculate_text_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# Load models
cohere_embedder = CohereEmbeddings()  # Semantic
# sim_model = SentenceTransformer("all-MiniLM-L6-v2") 

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set.")

DEFAULT_COLLECTION_NAME = "test_collection"
# Milvus
vectorstore = CohereMilvusRetriever(collection_name=DEFAULT_COLLECTION_NAME, embeddings=cohere_embedder)


def to_vector_store(pdf_path=None):
    logger.info("▶ Starting hybrid RAG pipeline (Milvus + Cohere + Hybrid Retrieval)...")

    documents = []
    if pdf_path and os.path.exists(pdf_path):
        logger.info("📄 Chunking PDF...")
        documents = chunk_pdf(pdf_path)
        # print(documents)

    if not documents:
        logger.error("❌ No valid documents loaded. Aborting pipeline.")
        return None, None
    # Insert documents into Milvus
    logger.info("💾 Inserting into Milvus...")
    docs = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    'page': chunk.metadata['page'],
                    # 'image_path': chunk.metadata.get('image_paths', []),
                    # 'reference_path': f"reference_imgs/reference_page_{chunk.metadata['page']}.png",
                }
            )
            for chunk in documents
        ]  # Debugging line
    vectorstore.add_documents(docs)
   
def run_pipeline(
    pdf_path=None,
    csv_path=None,
    output_csv=None
):
    to_vector_store(pdf_path) #Todo Uncomment
    df = pd.read_csv(csv_path)

    logger.info("🔍 Processing queries...")
    # ground_truths = df['GROUND TRUTH'].tolist()
    retrieved_answers_k_1 = []
    retrieved_answers_k_2 = []
    similarity_scores_k_1_list = []
    similarity_scores_k_2_list = []
    distances_k_1_list = []
    distances_k_2_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row['QUESTION']).strip()
        ground_truth = str(row['GROUND TRUTH']).strip()
        if not question or question.lower() == 'nan':
            logger.warning(f"Skipping empty question at index {idx}")
            continue
        results = vectorstore.retrieve(question, top_k=2)    #TODO Hybrid Search
        if not results:
            logger.warning(f"No results found for question at index {idx}")
            continue
        # logger.info(f"results: {results}")
        retrieved_answer = results[0].get("text", "").strip()
        logger.info(f"Retrieved answer : {retrieved_answer}----- {results}")
        retrieved_answers_k_1.append(retrieved_answer)
        retrieved_answers_k_2.append(results[1].get("text", "").strip())
        # retrieved_answers_k_3.append(results[2].get("text", "").strip())

        # Calculate similarity scores
        similarity_scores_k_1, distance_k_1 = calculate_text_similarity(retrieved_answer, ground_truth)
        similarity_scores_k_2, distance_k_2 = calculate_text_similarity(retrieved_answers_k_2[-1], ground_truth)
        # similarity_scores_k_3, distance_k_3 = calculate_text_similarity(retrieved_answers_k_3[-1], ground_truth)
        logger.info(f"Similarity K=1: {similarity_scores_k_1}, Distance K=1: {distance_k_1}")
        # TYPE
        logger.info(f"type of similarity_scores_k_1: {type(similarity_scores_k_1.item())}")
        logger.info(f"type of distance_k_1: {type(distance_k_1.item())}")

        similarity_scores_k_1_list.append(similarity_scores_k_1.item())
        similarity_scores_k_2_list.append(similarity_scores_k_2.item())
        # similarity_scores_k_3.append(similarity_scores_k_3[0][0])
        distances_k_1_list.append(distance_k_1.item())
        distances_k_2_list.append(distance_k_2.item())
        # distances_k_3.append(distance_k_3[0][0])

    # Create a DataFrame for the results
    results_df = pd.DataFrame({
        'QUESTION': df['QUESTION'],
        'GROUND TRUTH': df['GROUND TRUTH'],
        'RETRIEVED ANSWER K=1': retrieved_answers_k_1,
        'RETRIEVED ANSWER K=2': retrieved_answers_k_2,
        # 'RETRIEVED ANSWER K=3': retrieved_answers_k_3, 
        'SIMILARITY SCORE K=1': similarity_scores_k_1_list,
        'SIMILARITY SCORE K=2': similarity_scores_k_2_list,
        # 'SIMILARITY SCORE K=3': similarity_scores_k_3,
        'DISTANCE K=1': distances_k_1_list,
        'DISTANCE K=2': distances_k_2_list,
        # 'DISTANCE K=3': distances_k_3,
    })

    # store results
    results_df.to_csv(output_csv, index=False)
    

if __name__ == "__main__":
    pdf_path = "/mnt/c/Users/sanja/Downloads/BL9000-Owners-Manual.pdf"   
    csv_path = r"/mnt/c/Users/sanja/Downloads/hybrid search DATASET2 - DATASET.csv"   
    logger.info("into run.py")
    run_pipeline(
        pdf_path=pdf_path,
        csv_path=csv_path,
        output_csv="similarity_comparision.csv")
