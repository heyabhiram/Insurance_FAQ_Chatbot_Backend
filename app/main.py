from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.chat import router as chat_router
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    Verifies vector database indexing on startup.
    """
    logger.info("Initializing Vehicle Insurance Chatbot Backend...")
    try:
        faq_count = retrieval_service.count()
        if faq_count == 0:
            logger.info("No FAQs detected in ChromaDB. Ingesting initial dataset...")
            indexed = retrieval_service.ingest_faqs()
            logger.info(f"Ingestion complete: {indexed} FAQs loaded.")
        else:
            logger.info(f"ChromaDB ready with {faq_count} FAQs.")
    except Exception as e:
        logger.error(f"Error during initial FAQ indexing check: {e}")

    yield
    logger.info("Shutting down Vehicle Insurance Chatbot Backend.")

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade, beginner-friendly AI backend for Vehicle Insurance FAQs. "
        "Powered by FastAPI, NLP Intent Recognition, Sentence Transformers embeddings, "
        "ChromaDB semantic search, and TCS GenAI Lab LLM."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes under /api/v1
app.include_router(chat_router, prefix="/api/v1")

# Health Check Endpoint
@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Returns backend service health, database status, and LLM configuration details."
)
async def health_check():
    faq_count = 0
    chroma_status = "error"
    try:
        faq_count = retrieval_service.count()
        chroma_status = "ready"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")

    llm_connected = bool(settings.TCS_API_KEY and settings.TCS_API_KEY.strip())

    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "chroma_db_status": chroma_status,
        "faq_indexed_count": faq_count,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_configured": llm_connected,
        "llm_model": settings.TCS_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )

