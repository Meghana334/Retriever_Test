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
# from your_rag_module import run_pipeline, get_search_type_name, logger

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
    search_type: int
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




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "🚀 Multi-Modal RAG Pipeline API is running"}

