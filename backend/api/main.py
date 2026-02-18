"""
FastAPI application for Arabic Customer Service System
Provides REST API with Swagger documentation
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager

from backend.api.schemas import (
    QueryRequest,
    WorkflowResponse,
    DocumentResponse,
    HealthResponse,
    StatsResponse
)
from backend.orchestrator.workflow import MultiAgentWorkflow
from backend.data.knowledge_base_manager import KnowledgeBaseManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global workflow instance
workflow = None
kb_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global workflow, kb_manager
    
    logger.info("🚀 Starting Arabic Customer Service API...")
    
    # Initialize knowledge base
    logger.info("📚 Initializing knowledge base...")
    kb_manager = KnowledgeBaseManager()
    kb_stats = kb_manager.get_stats()
    logger.info(f"✅ Knowledge base loaded: {kb_stats['total_documents']} documents")
    
    # Initialize workflow
    logger.info("🔧 Initializing multi-agent workflow...")
    workflow = MultiAgentWorkflow()
    logger.info("✅ Workflow initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Arabic Customer Service API",
    description="""
    Multi-Agent Customer Service System with Arabic Support
    
    ## Features
    - 🤖 **Agent 1**: Intent Classification (6 categories)
    - 📚 **Agent 2**: Knowledge Retrieval (70 FAQ entries)
    - 🔄 **LangGraph**: Intelligent workflow orchestration
    - 🌐 **Arabic**: Full Arabic language support
    
    ## Workflow
    1. **Intent Classification**: Classify customer query
    2. **Knowledge Retrieval**: Find relevant FAQ entries
    3. **Response Generation**: Generate helpful response
    4. **Human Handoff**: Escalate when needed
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Arabic Customer Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /query": "Process customer query",
            "GET /health": "Health check",
            "GET /stats": "Knowledge base statistics"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and all agents are ready"
)
async def health_check():
    """Health check endpoint"""
    
    global workflow, kb_manager
    
    if not workflow or not kb_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not initialized"
        )
    
    kb_stats = kb_manager.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents_initialized=True,
        knowledge_base_documents=kb_stats['total_documents'],
        embedding_model=kb_stats['embedding_model']
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Knowledge base statistics",
    description="Get statistics about the knowledge base"
)
async def get_stats():
    """Get knowledge base statistics"""
    
    global kb_manager
    
    if not kb_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base not initialized"
        )
    
    stats = kb_manager.get_stats()
    
    return StatsResponse(
        total_documents=stats['total_documents'],
        embedding_model=stats['embedding_model'],
        categories=stats['categories'],
        collection_name=stats['collection_name']
    )


@app.post(
    "/query",
    response_model=WorkflowResponse,
    summary="Process customer query",
    description="""
    Process a customer query through the multi-agent workflow.
    
    The workflow will:
    1. Classify the intent (Agent 1)
    2. Retrieve relevant FAQs (Agent 2)
    3. Generate a response (Agent 3 - placeholder)
    4. Decide on human handoff if needed
    
    Returns the complete workflow result including:
    - Classified intent and confidence
    - Retrieved documents
    - Generated response
    - Execution timing
    """,
    responses={
        200: {
            "description": "Query processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "user_query": "ما هي أسعار الباقات المتاحة؟",
                        "intent": "product_inquiry",
                        "confidence": 0.95,
                        "sentiment": "neutral",
                        "requires_human": False,
                        "retrieved_documents": [
                            {
                                "question": "ما هي أسعار الباقات المتاحة؟",
                                "content": "نقدم ثلاث باقات...",
                                "relevance_score": 0.92,
                                "category": "product_inquiry",
                                "subcategory": "pricing"
                            }
                        ],
                        "response": "نقدم ثلاث باقات رئيسية...",
                        "workflow_status": "completed",
                        "total_time_ms": 1456,
                        "search_time_ms": 145
                    }
                }
            }
        },
        400: {"description": "Invalid request"},
        503: {"description": "Service unavailable"}
    }
)
async def process_query(request: QueryRequest):
    """
    Process a customer query through the multi-agent workflow
    
    Args:
        request: QueryRequest with customer query
        
    Returns:
        WorkflowResponse with complete workflow results
    """
    
    global workflow
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow not initialized"
        )
    
    try:
        # Process query through workflow
        result = await workflow.process_query(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        # Convert documents to response format
        documents = [
            DocumentResponse(
                question=doc['question'],
                content=doc['content'],
                relevance_score=doc['relevance_score'],
                category=doc['category'],
                subcategory=doc.get('subcategory')
            )
            for doc in result.get('retrieved_documents', [])
        ]
        
        # Build response
        response = WorkflowResponse(
            user_query=result['user_query'],
            conversation_id=result.get('conversation_id'),
            intent=result['intent'],
            confidence=result['confidence'],
            sentiment=result['sentiment'],
            requires_human=result['requires_human'],
            retrieved_documents=documents,
            response=result['response'],
            workflow_status=result['workflow_status'],
            total_time_ms=result['total_time_ms'],
            search_time_ms=result.get('search_time_ms', 0),
            error=result.get('error')
        )
        
        logger.info(
            f"✅ Query processed: {request.query[:50]}... | "
            f"Intent: {result['intent']} | "
            f"Time: {result['total_time_ms']:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )