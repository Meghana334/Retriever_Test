import os
import json
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import logging
from typing import List, Dict

from langchain.schema import Document
from search_modules import (
    HybridSearchModule,
    HybridSearchNoRerankModule,
    SemanticSearchModule,
    KeywordSearchModule,
)
from preprocessor import CohereEmbeddings, DocumentProcessor
from embed_utils import calculate_text_similarity

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set.")

DEFAULT_COLLECTION_NAME = "rag_documents"


def create_output_directories(base_output_dir: str = "output"):
    """Create organized output directory structure"""
    directories = {
        'base': base_output_dir,
        'json': os.path.join(base_output_dir, 'JSON'),
        'csv': os.path.join(base_output_dir, 'CSV'),
        'xlsx': os.path.join(base_output_dir, 'XLSX')
    }

    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"📁 Created/verified directory: {dir_path}")

    return directories


class RAGPipeline:
    """Multi-modal RAG Pipeline with different search strategies"""

    def __init__(
            self,
            collection_name: str = DEFAULT_COLLECTION_NAME,
            milvus_host: str = "localhost",
            milvus_port: str = "19530",
            output_base_dir: str = "output",
    ):

        self.doc_processor = DocumentProcessor(
            embedding=CohereEmbeddings(COHERE_API_KEY)
        )

        self.collection_name = collection_name
        self.cohere_embedder = CohereEmbeddings(COHERE_API_KEY)
        self.documents = []
        self.output_dirs = create_output_directories(output_base_dir)

        # Initialize all search modules
        self.hybrid_search = HybridSearchModule(
            collection_name=collection_name,
            embeddings=self.cohere_embedder,
            host=milvus_host,
            port=milvus_port,
        )

        self.hybrid_no_rerank = HybridSearchNoRerankModule(
            collection_name=f"{collection_name}_no_rerank",
            embeddings=self.cohere_embedder,
            host=milvus_host,
            port=milvus_port,
        )

        self.semantic_search = SemanticSearchModule(
            collection_name=f"{collection_name}_semantic",
            embeddings=self.cohere_embedder,
            host=milvus_host,
            port=milvus_port,
        )

        self.keyword_search = KeywordSearchModule()

        logger.info(f"✅ Initialized RAG Pipeline with collection: {collection_name}")
        logger.info(f"📂 Output directories: {self.output_dirs}")

    def load_documents(self, pdf_path: str):
        """Load PDF documents and initialize all search modules"""
        logger.info("📄 Loading and processing documents...")

        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"❌ PDF file not found: {pdf_path}")
            return

        documents = self.doc_processor.load_pdf(pdf_path)
        logger.info(f"documents: {len(documents)}")
        # logger.info(documents)
        self.documents = self.doc_processor.preprocess_documents(documents)

        if not self.documents:
            logger.error("❌ No valid documents loaded. Aborting pipeline.")
            return

        logger.info(f"📚 Loaded {len(self.documents)} documents")

        logger.info("🔄 Initializing search modules...")
        self.hybrid_search.add_documents(self.documents)
        self.hybrid_no_rerank.add_documents(self.documents)
        self.semantic_search.add_documents(self.documents)
        self.keyword_search.fit_documents([doc.page_content for doc in self.documents])
        logger.info("✅ All search modules initialized successfully")

    def search(self, query: str, search_type: int, top_k: int = 3) -> List[Dict]:
        """Perform search based on selected type"""
        if search_type == 1:
            return self.hybrid_search.search(query, top_k)
        elif search_type == 2:
            return self.hybrid_no_rerank.search(query, top_k)
        elif search_type == 3:
            return self.semantic_search.search(query, top_k)
        elif search_type == 4:
            return self.keyword_search.search(query, self.documents, top_k)
        else:
            logger.error(f"Invalid search type: {search_type}")
            return []

    def get_output_paths(self, search_type: int):
        """Generate organized output file paths"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_suffix = f"type_{search_type}"
        base_filename = f"rag_results_{search_suffix}_{timestamp}"

        return {
            'csv': os.path.join(self.output_dirs['csv'], f"{base_filename}.csv"),
            'json': os.path.join(self.output_dirs['json'], f"rag_metadata_{search_suffix}_{timestamp}.json"),
            'xlsx': os.path.join(self.output_dirs['xlsx'], f"{base_filename}.xlsx")
        }


def get_search_type_name(search_type: int) -> str:
    """Get human-readable search type name"""
    search_names = {
        1: "Hybrid Search (with reranking)",
        2: "Hybrid Search (without reranking)",
        3: "Semantic Search",
        4: "Keyword Search (BM25)",
    }
    return search_names.get(search_type, "Unknown")


def run_pipeline(
        pdf_path: str = None,
        csv_path: str = None,
        search_type: int = 1,
        output_base_dir: str = "output",
):
    """Run the complete RAG pipeline with specified search type and organized output"""

    start_time = time.time()
    search_type_name = get_search_type_name(search_type)

    logger.info(f"🚀 Starting RAG Pipeline with {search_type_name}")
    pipeline = RAGPipeline(output_base_dir=output_base_dir)

    # Get organized output paths
    output_paths = pipeline.get_output_paths(search_type)

    if pdf_path:
        doc_load_start = time.time()
        pipeline.load_documents(pdf_path)
        doc_load_time = time.time() - doc_load_start
        logger.info(f"⏱️ Document loading time: {doc_load_time:.2f} seconds")

    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        return

    logger.info(f"📊 Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    print(df)

    df.columns = df.columns.str.strip().str.upper()
    logger.info(f"📈 Loaded {len(df)} questions from CSV")

    all_metadata = []
    retrieved_answers_k_1, retrieved_answers_k_2, retrieved_answers_k_3 = [], [], []
    (
        similarity_scores_k_1_list,
        similarity_scores_k_2_list,
        similarity_scores_k_3_list,
    ) = ([], [], [])
    distances_k_1_list, distances_k_2_list, distances_k_3_list = [], [], []
    binary_classification_k_1, binary_classification_k_2, binary_classification_k_3 = (
        [],
        [],
        [],
    )

    total_query_time = 0
    processed_questions = 0

    for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing with {search_type_name}"
    ):
        question_start_time = time.time()
        question = str(row["QUESTION"]).strip()
        ground_truth = str(row["GROUND TRUTH"]).strip()

        if not question or question.lower() == "nan":
            retrieved_answers_k_1.append("")
            retrieved_answers_k_2.append("")
            retrieved_answers_k_3.append("")
            similarity_scores_k_1_list.append(0.0)
            similarity_scores_k_2_list.append(0.0)
            similarity_scores_k_3_list.append(0.0)
            distances_k_1_list.append(1.0)
            distances_k_2_list.append(1.0)
            distances_k_3_list.append(1.0)
            binary_classification_k_1.append(False)
            binary_classification_k_2.append(False)
            binary_classification_k_3.append(False)
            continue

        search_start_time = time.time()
        results = pipeline.search(question, search_type, top_k=3)
        search_time = time.time() - search_start_time

        while len(results) < 3:
            results.append(
                {
                    "text": "No answer found",
                    "metadata": {"page": "N/A", "chunk": "N/A"},
                    "similarity": 0.0,
                    "doc_id": -1,
                }
            )

        answers_meta = []
        for k in range(3):
            result = results[k]
            answer = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            page = metadata.get("page", "N/A")
            chunk = metadata.get("chunk", "N/A")
            sim_score, distance = calculate_text_similarity(answer, ground_truth)
            sim_score_float = (
                float(sim_score[0][0])
                if hasattr(sim_score, "shape")
                else float(sim_score)
            )
            distance_float = (
                float(distance[0][0]) if hasattr(distance, "shape") else float(distance)
            )
            binary_class = sim_score_float >= 0.6
            answers_meta.append(
                {
                    "answer": answer,
                    "page": page,
                    "chunk": chunk,
                    "similarity_score": sim_score_float,
                    "distance": distance_float,
                }
            )

            if k == 0:
                retrieved_answers_k_1.append(answer)
                similarity_scores_k_1_list.append(sim_score_float)
                distances_k_1_list.append(distance_float)
                binary_classification_k_1.append(binary_class)
            elif k == 1:
                retrieved_answers_k_2.append(answer)
                similarity_scores_k_2_list.append(sim_score_float)
                distances_k_2_list.append(distance_float)
                binary_classification_k_2.append(binary_class)
            elif k == 2:
                retrieved_answers_k_3.append(answer)
                similarity_scores_k_3_list.append(sim_score_float)
                distances_k_3_list.append(distance_float)
                binary_classification_k_3.append(binary_class)

        all_metadata.append({"question": question, "answers": answers_meta})
        total_query_time += search_time
        processed_questions += 1
        question_time = time.time() - question_start_time
        logger.info(
            f"⏱️ Question {idx + 1}: Total={question_time:.3f}s, Search={search_time:.3f}s"
        )

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "QUESTION": df["QUESTION"],
            "GROUND TRUTH": df["GROUND TRUTH"],
            "RETRIEVED ANSWER 1": retrieved_answers_k_1,
            "RETRIEVED ANSWER 2": retrieved_answers_k_2,
            "RETRIEVED ANSWER 3": retrieved_answers_k_3,
            "SIMILARITY SCORE 1": similarity_scores_k_1_list,
            "SIMILARITY SCORE 2": similarity_scores_k_2_list,
            "SIMILARITY SCORE 3": similarity_scores_k_3_list,
            "DISTANCE 1": distances_k_1_list,
            "DISTANCE 2": distances_k_2_list,
            "DISTANCE 3": distances_k_3_list,
            "BINARY CLASSIFICATION 1": binary_classification_k_1,
            "BINARY CLASSIFICATION 2": binary_classification_k_2,
            "BINARY CLASSIFICATION 3": binary_classification_k_3,
        }
    )

    # Add Combined Classification column
    # True if ANY of the three binary classifications is True
    results_df["COMBINED CLASSIFICATION"] = (
            results_df["BINARY CLASSIFICATION 1"] |
            results_df["BINARY CLASSIFICATION 2"] |
            results_df["BINARY CLASSIFICATION 3"]
    )

    # Save CSV file to organized structure
    results_df.to_csv(output_paths['csv'], index=False)
    logger.info(f"📁 CSV results saved to {output_paths['csv']}")

    # Save JSON metadata to organized structure
    with open(output_paths['json'], "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    logger.info(f"📁 Metadata JSON saved to {output_paths['json']}")

    # Calculate performance metrics
    total_time = time.time() - start_time
    avg_query_time = (
        total_query_time / processed_questions if processed_questions > 0 else 0
    )

    performance_summary = pd.DataFrame(
        [
            ["Search Type", search_type_name],
            ["Total Processing Time (s)", f"{total_time:.2f}"],
            ["Total Query Time (s)", f"{total_query_time:.2f}"],
            ["Average Query Time (s)", f"{avg_query_time:.3f}"],
            ["Questions Processed", processed_questions],
        ],
        columns=["Metric", "Value"],
    )

    # Calculate accuracy metrics (including combined classification)
    accuracy_rows = []
    for k in [1, 2, 3]:
        scores = results_df[f"SIMILARITY SCORE {k}"]
        binary_class = results_df[f"BINARY CLASSIFICATION {k}"]
        print("=======")
        print(binary_class)
        print(binary_class.sum())
        print(len(binary_class))
        print("=======")

        valid_scores = scores[scores > 0]
        accuracy = binary_class.sum() / len(binary_class) * 100
        if len(valid_scores) > 0:
            accuracy_rows.append(
                [
                    f"K={k}",
                    f"{valid_scores.mean():.4f}",
                    f"{valid_scores.max():.4f}",
                    f"{valid_scores.min():.4f}",
                    f"{accuracy:.2f}%",
                ]
            )

    # Add combined classification accuracy
    combined_accuracy = results_df["COMBINED CLASSIFICATION"].sum() / len(results_df) * 100
    accuracy_rows.append(
        [
            "Combined",
            "N/A",
            "N/A",
            "N/A",
            f"{combined_accuracy:.2f}%"
        ]
    )

    accuracy_df = pd.DataFrame(
        accuracy_rows, columns=["K", "Avg Similarity", "Max", "Min", "Accuracy ≥0.6"]
    )

    # Save Excel file with multiple sheets to organized structure
    with pd.ExcelWriter(output_paths['xlsx'], engine="xlsxwriter") as writer:
        results_df.to_excel(writer, sheet_name="Results", index=False)
        performance_summary.to_excel(
            writer, sheet_name="Performance Summary", index=False
        )
        accuracy_df.to_excel(writer, sheet_name="Accuracy Summary", index=False)
    logger.info(f"📁 Excel results saved to {output_paths['xlsx']}")

    # Summary of all output files
    logger.info("=" * 60)
    logger.info("📋 OUTPUT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"🗂️  Output Directory Structure:")
    logger.info(f"   📂 {pipeline.output_dirs['base']}/")
    logger.info(f"   ├── 📂 CSV/")
    logger.info(f"   │   └── 📄 {os.path.basename(output_paths['csv'])}")
    logger.info(f"   ├── 📂 JSON/")
    logger.info(f"   │   └── 📄 {os.path.basename(output_paths['json'])}")
    logger.info(f"   └── 📂 XLSX/")
    logger.info(f"       └── 📄 {os.path.basename(output_paths['xlsx'])}")
    logger.info("=" * 60)

    # Log combined classification summary
    logger.info(f"🎯 Combined Classification Summary:")
    logger.info(f"   Total questions: {len(results_df)}")
    logger.info(f"   Successfully classified (any K): {results_df['COMBINED CLASSIFICATION'].sum()}")
    logger.info(f"   Combined accuracy: {combined_accuracy:.2f}%")

    return results_df, output_paths


def main():
    print("🚀 Multi-Modal RAG Pipeline")
    print("=" * 50)
    print("Search Options:")
    print("1. Hybrid Search (with reranking)")
    print("2. Hybrid Search (without reranking)")
    print("3. Semantic Search")
    print("4. Keyword Search (BM25)")
    print("=" * 50)

    try:
        search_type = int(input("Enter search type (1-4): "))
        if search_type not in [1, 2, 3, 4]:
            raise ValueError("Invalid search type")
    except ValueError:
        logger.error("❌ Invalid input. Please enter a number between 1-4.")
        return

    # Option to specify custom output directory
    try:
        output_dir = input("Enter output directory name (default: 'output'): ").strip()
        if not output_dir:
            output_dir = "output"
    except:
        output_dir = "output"

    PDF_PATH = "/home/meghana/Downloads/BL9000-Owners-Manual.pdf"
    CSV_PATH = r"/home/meghana/Downloads/questions and ground truths.csv"

    search_type_name = get_search_type_name(search_type)
    logger.info(f"🎯 Selected: {search_type_name}")
    logger.info(f"📂 Output directory: {output_dir}")

    try:
        if CSV_PATH and os.path.exists(CSV_PATH):
            logger.info(f"📊 Running {search_type_name} pipeline...")
            results_df, output_paths = run_pipeline(
                pdf_path=PDF_PATH,
                csv_path=CSV_PATH,
                search_type=search_type,
                output_base_dir=output_dir,
            )
            logger.info("🎉 RAG pipeline completed successfully!")
            logger.info(f"📈 CSV Results: {output_paths['csv']}")
            logger.info(f"📋 JSON Metadata: {output_paths['json']}")
            logger.info(f"📊 Excel Report: {output_paths['xlsx']}")
        else:
            logger.error(f"❌ CSV file not found: {CSV_PATH}")
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
