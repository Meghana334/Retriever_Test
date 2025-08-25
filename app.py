from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
import asyncio
from pathlib import Path
import tempfile
import shutil
from main import run_pipeline
# Import your existing functions (assuming they're in the same file or imported)
from jiwer import wer
import pandas as pd

app = FastAPI(
    title="Multi-Modal RAG Pipeline API",
    description="API for running Multi-Modal RAG Pipeline with different search options",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging (preserve your existing logger)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request models
class PipelineRequest(BaseModel):
    search_type: int = 1
    output_dir: Optional[str] = "output"
    pdf_path: Optional[str] = None
    csv_path: Optional[str] = None


class PipelineResponse(BaseModel):
    success: bool
    message: str
    search_type_name: str
    output_paths: Optional[Dict[str, str]] = None
    error: Optional[str] = None


# Your existing function (assuming it's available)
def get_search_type_name(search_type: int) -> str:
    """Get human-readable name for search type"""
    search_names = {
        1: "Hybrid Search (with reranking)",
        2: "Hybrid Search (without reranking)",
        3: "Semantic Search",
        4: "Keyword Search (BM25)"
    }
    return search_names.get(search_type, "Unknown")


# Placeholder for your run_pipeline function



@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Multi-Modal RAG Pipeline API",
        "version": "1.0.0",
        "search_options": {
            "1": "Hybrid Search (with reranking)",
            "2": "Hybrid Search (without reranking)",
            "3": "Semantic Search",
            "4": "Keyword Search (BM25)"
        }
    }


@app.get("/search-options")
async def get_search_options():
    """Get available search options"""
    logger.info("📋 Retrieving search options")
    return {
        "search_options": {
            "1": "Hybrid Search (with reranking)",
            "2": "Hybrid Search (without reranking)",
            "3": "Semantic Search",
            "4": "Keyword Search (BM25)"
        }
    }

@app.post("/pipeline")
async def run_pipeline_with_files(
        search_type: int = Form(...),
        output_dir: str = Form("output"),
        pdf_file: UploadFile = File(...),
        csv_file: UploadFile = File(...)
):
    """
    Run the RAG pipeline with uploaded files
    """
    print("🚀 Multi-Modal RAG Pipeline API Request (with file uploads)")
    print("=" * 50)

    try:
        # Validate search type
        if search_type not in [1, 2, 3, 4]:
            logger.error("❌ Invalid search type. Please provide a number between 1-4.")
            raise HTTPException(status_code=400, detail="Invalid search type. Must be between 1-4.")

        search_type_name = get_search_type_name(search_type)
        logger.info(f"🎯 Selected: {search_type_name}")
        logger.info(f"📂 Output directory: {output_dir}")
        logger.info(f"📄 PDF File: {pdf_file.filename}")
        logger.info(f"📊 CSV File: {csv_file.filename}")

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded PDF
            pdf_path = os.path.join(temp_dir, pdf_file.filename)
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)

            # Save uploaded CSV
            csv_path = os.path.join(temp_dir, csv_file.filename)
            with open(csv_path, "wb") as buffer:
                shutil.copyfileobj(csv_file.file, buffer)

            # Run the pipeline
            logger.info(f"📊 Running {search_type_name} pipeline...")
            results_df, output_paths = run_pipeline(
                pdf_path=pdf_path,
                csv_path=csv_path,
                search_type=search_type,
                output_base_dir=output_dir,
            )

            logger.info("🎉 RAG pipeline completed successfully!")
            logger.info(f"📈 CSV Results: {output_paths['csv']}")
            logger.info(f"📋 JSON Metadata: {output_paths['json']}")
            logger.info(f"📊 Excel Report: {output_paths['xlsx']}")

            return PipelineResponse(
                success=True,
                message="RAG pipeline completed successfully!",
                search_type_name=search_type_name,
                output_paths=output_paths
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/wer")
def calculate_wer(
        csv_file: UploadFile = File(...),
):
    csv_path = os.path.join('output/XLSX', csv_file.filename)

    # Save uploaded file
    with open(csv_path, "wb") as buffer:
        buffer.write(csv_file.file.read())

    logger.info(f"Saved CSV/Excel file: {csv_path}")

    # Read Excel file
    df = pd.read_excel(csv_path)

    # Initialize new columns
    df['wer_1'] = 0.0
    df['wer_2'] = 0.0
    df['wer_3'] = 0.0
    df['wer_best'] = 0.0

    # Iterate through rows
    for idx, row in df.iterrows():
        gt = str(row['GROUND TRUTH'])

        # Compute WERs
        w1 = wer(gt, str(row['RETRIEVED ANSWER 1']))
        w2 = wer(gt, str(row['RETRIEVED ANSWER 2']))
        w3 = wer(gt, str(row['RETRIEVED ANSWER 3']))

        # Store in dataframe
        df.at[idx, 'wer_1'] = w1
        df.at[idx, 'wer_2'] = w2
        df.at[idx, 'wer_3'] = w3
        df.at[idx, 'wer_best'] = min(w1, w2, w3)

        # Log details
        logger.info(
            f"Row {idx} | WER_1: {w1:.4f}, WER_2: {w2:.4f}, WER_3: {w3:.4f}, Best: {df.at[idx, 'wer_best']:.4f}"
        )

    # Drop intermediate WER columns
    # df.drop(['wer_1', 'wer_2', 'wer_3'], axis=1, inplace=True)

    # Save updated CSV
    df.to_csv(csv_path, index=False)

    # Compute average best WER
    avg_wer = df['wer_best'].mean()
    logger.info(f"Average Best WER: {avg_wer:.4f}")

    return {'wer': avg_wer}





@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "🚀 Multi-Modal RAG Pipeline API is running"}

