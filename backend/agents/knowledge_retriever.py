"""
Knowledge Retriever Agent (Agent 2)
Retrieves relevant FAQ entries from vector database
"""

import time
from typing import Dict, Any, List, Optional
from groq import Groq

from .base_agent import BaseAgent
from backend.data.knowledge_base_manager import KnowledgeBaseManager
from backend.config.settings import GROQ_API_KEY, RETRIEVAL_CONFIG
from backend.utils.logger import get_logger, format_arabic_for_terminal

logger = get_logger(__name__)


class KnowledgeRetrieverAgent(BaseAgent):
    """
    Agent 2: Knowledge Retrieval
    
    Responsibilities:
    - Search vector database for relevant FAQ entries
    - Filter by category (from Agent 1's intent)
    - Rank results by relevance
    - Return top 3-5 most relevant documents
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        n_results: Optional[int] = None
    ):
        super().__init__(name="KnowledgeRetriever", model_name=model_name)
        
        # Use n_results from config if not provided
        self.n_results = n_results or RETRIEVAL_CONFIG["n_results"]
        
        # Initialize knowledge base
        logger.info(f"🔧 Initializing knowledge base...")
        self.kb = KnowledgeBaseManager()
        
        # Initialize Groq client (for future reranking if needed)
        self.client = Groq(api_key=GROQ_API_KEY)
        
        kb_stats = self.kb.get_stats()
        
        logger.info(f"✅ KnowledgeRetrieverAgent initialized")
        logger.info(f"   Documents in KB: {kb_stats['total_documents']}")
        logger.info(f"   Embedding Model: {kb_stats['embedding_model']}")
        logger.info(f"   Results per query: {self.n_results}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant documents from knowledge base
        
        Args:
            input_data: {
                "user_query": str,
                "intent": str,  # From Agent 1
                "confidence": float  # From Agent 1 (optional)
            }
            
        Returns:
            {
                "documents": List[Dict],  # Top N relevant docs
                "search_time_ms": float,
                "total_found": int,
                "retrieval_method": str,
                "category_filter": str
            }
        """
        
        user_query = input_data.get("user_query", "")
        intent = input_data.get("intent", "")
        
        if not user_query:
            return self._error_response("Empty query provided")
        
        query_display = format_arabic_for_terminal(user_query, max_length=60)
        logger.info(f"🔍 Retrieving docs for: {query_display}")
        logger.info(f"   Intent: {intent}")
        
        # Map intent to category
        category_filter = self._map_intent_to_category(intent)
        
        # Search vector database
        start_time = time.time()
        
        try:
            results = self.kb.search(
                query=user_query,
                n_results=self.n_results,
                category_filter=category_filter
            )
            
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Process results
            documents = self._process_results(results)
            
            logger.info(
                f"✅ Retrieved {len(documents)} docs | "
                f"category: {category_filter or 'all'} | "
                f"{search_time:.0f}ms"
            )
            
            # Log top result for debugging
            if documents:
                top_doc = documents[0]
                top_question = format_arabic_for_terminal(top_doc['question'], 40)
                logger.debug(f"Top result: {top_question} (score: {top_doc['relevance_score']:.2f})")
            else:
                logger.warning("⚠️  No documents found!")
            
            return {
                "documents": documents,
                "search_time_ms": search_time,
                "total_found": len(documents),
                "retrieval_method": "vector_search",
                "category_filter": category_filter,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {str(e)}")
            return self._error_response(str(e))
    
    def _map_intent_to_category(self, intent: str) -> Optional[str]:
        """Map intent to knowledge base category"""
        
        intent_to_category = {
            "product_inquiry": "product_inquiry",
            "technical_support": "technical_support",
            "billing_question": "billing_question",
            "complaint": None,  # Search all categories
            "general_question": "general_question",
            "escalation_needed": None  # Search all categories
        }
        
        return intent_to_category.get(intent)
    
    def _process_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Process and format search results"""
        
        documents = []
        
        if not results['documents'][0]:
            return documents
        
        for i in range(len(results['documents'][0])):
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Convert distance to similarity score (0-1, higher is better)
            # ChromaDB uses L2 distance, so we convert it
            relevance_score = 1 - min(distance, 1.0)  # Clamp to [0, 1]
            
            # Extract question and answer from document
            parts = document.split('\n\n', 1)
            question = parts[0] if len(parts) > 0 else ""
            answer = parts[1] if len(parts) > 1 else document
            
            documents.append({
                'content': answer,
                'question': question,
                'full_text': document,
                'relevance_score': float(relevance_score),
                'distance': float(distance),
                'category': metadata.get('category', ''),
                'subcategory': metadata.get('subcategory', ''),
                'priority': metadata.get('priority', 'medium'),
                'keywords': metadata.get('keywords', '').split(',') if metadata.get('keywords') else [],
                'metadata': metadata
            })
        
        # Sort by relevance score (highest first)
        documents.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return documents
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output format"""
        
        required_fields = ['documents', 'search_time_ms', 'total_found']
        
        if not all(field in output for field in required_fields):
            logger.warning(f"Missing required fields: {required_fields}")
            return False
        
        if not isinstance(output['documents'], list):
            logger.warning("Documents field is not a list")
            return False
        
        return True
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Return error response"""
        return {
            "documents": [],
            "search_time_ms": 0,
            "total_found": 0,
            "retrieval_method": "error",
            "category_filter": None,
            "error": error_message,
            "model": self.model_name
        }
    
    # ========== SYNCHRONOUS WRAPPER FOR BENCHMARKING ==========
    
    def retrieve(self, user_query: str, intent: str = "") -> Dict[str, Any]:
        """
        Synchronous retrieval (for benchmarking)
        
        Args:
            user_query: User query in Arabic
            intent: Intent from Agent 1
            
        Returns:
            Retrieval result dictionary
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process({
                "user_query": user_query,
                "intent": intent
            })
        )