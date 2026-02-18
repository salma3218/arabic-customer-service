"""
LangGraph State Definition
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    State shared across all agents in the workflow
    """
    
    # Input
    user_query: str
    conversation_id: Optional[str]
    
    # Agent 1 Output
    intent: str
    confidence: float
    sentiment: str
    requires_human: bool
    classification_reasoning: str
    
    # Agent 2 Output
    retrieved_documents: List[Dict[str, Any]]
    retrieval_method: str
    search_time_ms: float
    
    # Agent 3 Output (placeholder for now)
    response: str
    
    # Metadata
    current_agent: str
    workflow_status: str
    error: Optional[str]
    total_time_ms: float