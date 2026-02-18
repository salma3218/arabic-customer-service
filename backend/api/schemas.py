"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    """Request schema for processing a customer query"""
    
    query: str = Field(
        ...,
        description="Customer query in Arabic",
        min_length=1,
        example="ما هي أسعار الباقات المتاحة؟"
    )
    
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking",
        example="conv_123456"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "ما هي أسعار الباقات المتاحة؟",
                "conversation_id": "conv_123456"
            }
        }


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class DocumentResponse(BaseModel):
    """Schema for a retrieved document"""
    
    question: str = Field(..., description="FAQ question")
    content: str = Field(..., description="FAQ answer")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    category: str = Field(..., description="Document category")
    subcategory: Optional[str] = Field(None, description="Document subcategory")


class WorkflowResponse(BaseModel):
    """Response schema for workflow execution"""
    
    # Input
    user_query: str = Field(..., description="Original user query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    
    # Agent 1 Output
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Classification confidence (0-1)")
    sentiment: str = Field(..., description="Detected sentiment")
    requires_human: bool = Field(..., description="Whether human handoff is needed")
    
    # Agent 2 Output
    retrieved_documents: List[DocumentResponse] = Field(
        default_factory=list,
        description="Retrieved FAQ documents"
    )
    
    # Response
    response: str = Field(..., description="Generated response")
    
    # Metadata
    workflow_status: str = Field(..., description="Workflow status")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    search_time_ms: float = Field(default=0, description="Search time in milliseconds")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_query": "ما هي أسعار الباقات المتاحة؟",
                "conversation_id": "conv_123456",
                "intent": "product_inquiry",
                "confidence": 0.95,
                "sentiment": "neutral",
                "requires_human": False,
                "retrieved_documents": [
                    {
                        "question": "ما هي أسعار الباقات المتاحة؟",
                        "content": "نقدم ثلاث باقات رئيسية...",
                        "relevance_score": 0.92,
                        "category": "product_inquiry",
                        "subcategory": "pricing"
                    }
                ],
                "response": "نقدم ثلاث باقات رئيسية: الباقة الأساسية بسعر 99 ريال شهرياً...",
                "workflow_status": "completed",
                "total_time_ms": 1456,
                "search_time_ms": 145
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    agents_initialized: bool = Field(..., description="Whether agents are ready")
    knowledge_base_documents: int = Field(..., description="Number of documents in KB")
    embedding_model: str = Field(..., description="Embedding model being used")


class StatsResponse(BaseModel):
    """Statistics response"""
    
    total_documents: int
    embedding_model: str
    categories: Dict[str, int]
    collection_name: str