"""
LangGraph Workflow for Multi-Agent System
Orchestrates Agents 1 and 2 with LLM-based reranking
"""

import time
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from groq import Groq

from backend.agents.intent_classifier import IntentClassifierAgent
from backend.agents.knowledge_retriever import KnowledgeRetrieverAgent
from .state import AgentState
from backend.config.settings import GROQ_API_KEY, AGENT_2_MODEL
from backend.utils.logger import get_logger, format_arabic_for_terminal

logger = get_logger(__name__)


class MultiAgentWorkflow:
    """
    Multi-Agent Customer Service Workflow
    
    Flow:
    1. Entry → Agent 1 (Intent Classification)
    2. Decision → Human Handoff OR Agent 2
    3. Agent 2 (Knowledge Retrieval) + LLM Reranking
    4. Agent 3 (Response Generation) - TODO
    5. END
    """
    
    def __init__(self):
        # Initialize agents
        logger.info("🔧 Initializing agents...")
        self.agent1 = IntentClassifierAgent()
        self.agent2 = KnowledgeRetrieverAgent()
        
        # Initialize Groq client for reranking
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.reranking_model = AGENT_2_MODEL
        
        # Build workflow
        self.app = self._build_workflow()
        
        logger.info("✅ Multi-Agent Workflow initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
        workflow.add_node("human_handoff", self._human_handoff_node)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "retrieve": "retrieve_knowledge",
                "human": "human_handoff"
            }
        )
        
        # Add edges to END
        workflow.add_edge("retrieve_knowledge", END)
        workflow.add_edge("human_handoff", END)
        
        # Compile
        return workflow.compile()
    
    async def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Node: Agent 1 - Intent Classification"""
        
        logger.info("🔹 Node: classify_intent")
        
        # Call Agent 1
        result = await self.agent1.process({"query": state["user_query"]})
        
        # Update state
        state["intent"] = result["intent"]
        state["confidence"] = result["confidence"]
        state["sentiment"] = result.get("sentiment", "neutral")
        state["requires_human"] = result.get("requires_human", False)
        state["classification_reasoning"] = result.get("reasoning", "")
        state["current_agent"] = "agent_1"
        
        return state
    
    async def _retrieve_knowledge_node(self, state: AgentState) -> AgentState:
        """
        Node: Agent 2 - Knowledge Retrieval + LLM Reranking
        
        ✅ UPDATED: Now includes intelligent document selection using LLM
        """
        
        logger.info("🔹 Node: retrieve_knowledge")
        
        # Call Agent 2
        result = await self.agent2.process({
            "user_query": state["user_query"],
            "intent": state["intent"],
            "confidence": state["confidence"]
        })
        
        # Update state
        state["retrieved_documents"] = result["documents"]
        state["retrieval_method"] = result.get("retrieval_method", "vector_search")
        state["search_time_ms"] = result.get("search_time_ms", 0)
        state["current_agent"] = "agent_2"
        state["workflow_status"] = "documents_retrieved"
        
        # ✅ IMPROVED: Use LLM to select best document
        if state["retrieved_documents"]:
            logger.info("🤖 Using LLM to select best document...")
            
            best_doc = await self._select_best_document_with_llm(
                query=state["user_query"],
                documents=state["retrieved_documents"]
            )
            
            if best_doc:
                state["response"] = best_doc.get("content", "لا توجد معلومات متاحة")
                
                # Log which document was selected
                question_preview = format_arabic_for_terminal(best_doc['question'], 50)
                logger.info(f"✅ Selected document: {question_preview}")
            else:
                # Fallback to first document
                logger.warning("⚠️  LLM reranking failed, using first document")
                state["response"] = state["retrieved_documents"][0].get("content", "لا توجد معلومات متاحة")
        else:
            state["response"] = "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك"
        
        return state
    
    async def _select_best_document_with_llm(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Use LLM to intelligently select the most relevant document
        
        This solves the problem where vector search ranks documents poorly.
        The LLM can understand context and select the truly relevant answer.
        
        Args:
            query: User's query in Arabic
            documents: List of retrieved documents (up to 10)
            
        Returns:
            The most relevant document, or None if selection fails
        """
        
        if not documents:
            return None
        
        # If only one document, return it
        if len(documents) == 1:
            return documents[0]
        
        try:
            # Build prompt with top documents (limit to 5 for context size)
            top_docs = documents[:10]
            
            docs_text = ""
            for i, doc in enumerate(top_docs, 1):
                # Truncate content to save tokens
                content_preview = doc['content'][:300] if len(doc['content']) > 300 else doc['content']
                docs_text += f"""
[الوثيقة {i}]
السؤال: {doc['question']}
الإجابة: {content_preview}
الفئة: {doc['category']}
درجة التطابق: {doc['relevance_score']:.2f}

"""
            
            # Create selection prompt
            prompt = f"""أنت مساعد ذكي متخصص في اختيار أفضل إجابة لأسئلة العملاء.

سؤال العميل: "{query}"

لديك {len(top_docs)} وثائق متاحة. اختر الوثيقة الأكثر صلة وملاءمة للإجابة على سؤال العميل.

الوثائق المتاحة:
{docs_text}

يجب عليك الرد فقط برقم الوثيقة الأكثر صلة (من 1 إلى {len(top_docs)}).
لا تكتب أي شيء آخر، فقط الرقم.

رقم الوثيقة الأفضل:"""
            
            # Call LLM for selection
            start_time = time.time()
            
            response = self.groq_client.chat.completions.create(
                model=self.reranking_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0  # Deterministic selection
            )
            
            reranking_time = (time.time() - start_time) * 1000
            
            # Extract selection
            answer = response.choices[0].message.content.strip()
            
            # Parse the number (handle various formats)
            import re
            numbers = re.findall(r'\d+', answer)
            
            if numbers:
                selected_num = int(numbers[0])
                
                # Validate range
                if 1 <= selected_num <= len(top_docs):
                    selected_doc = top_docs[selected_num - 1]
                    
                    logger.info(
                        f"🎯 LLM selected document #{selected_num} "
                        f"(original rank: {documents.index(selected_doc) + 1}) "
                        f"in {reranking_time:.0f}ms"
                    )
                    
                    return selected_doc
                else:
                    logger.warning(f"⚠️  LLM returned out-of-range number: {selected_num}")
            else:
                logger.warning(f"⚠️  Could not parse LLM response: {answer}")
        
        except Exception as e:
            logger.error(f"❌ LLM reranking failed: {str(e)}")
        
        # Fallback: return first document
        logger.info("📊 Falling back to first document")
        return documents[0]
    
    async def _human_handoff_node(self, state: AgentState) -> AgentState:
        """Node: Human Handoff"""
        
        logger.info("🔹 Node: human_handoff")
        
        state["current_agent"] = "human"
        state["workflow_status"] = "escalated_to_human"
        state["response"] = "سيتم تحويلك إلى موظف خدمة العملاء للمساعدة. الرجاء الانتظار..."
        state["retrieved_documents"] = []
        
        return state
    
    def _route_after_classification(self, state: AgentState) -> str:
        """
        Decision function: Route to Agent 2 or Human
        
        Routes to human if:
        - requires_human = True
        - Low confidence (< 0.70)
        - Complaint with very negative sentiment
        - Escalation needed intent
        """
        
        # Check explicit human requirement
        if state.get("requires_human", False):
            logger.info("🔀 Routing: human (explicit requirement)")
            return "human"
        
        # Check for escalation intent
        if state.get("intent") == "escalation_needed":
            logger.info("🔀 Routing: human (escalation intent)")
            return "human"
        
        # Check confidence threshold
        if state.get("confidence", 0) < 0.70:
            logger.info(f"🔀 Routing: human (low confidence: {state['confidence']:.2f})")
            return "human"
        
        # Check complaint + very negative sentiment
        if (state.get("intent") == "complaint" and 
            state.get("sentiment") == "very_negative"):
            logger.info("🔀 Routing: human (complaint + very negative sentiment)")
            return "human"
        
        # Continue to Agent 2
        logger.info("🔀 Routing: retrieve (continue workflow)")
        return "retrieve"
    
    async def process_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Process a user query through the workflow
        
        Args:
            query: User query in Arabic
            conversation_id: Optional conversation ID
            
        Returns:
            Final state with response
        """
        
        query_display = format_arabic_for_terminal(query, max_length=60)
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing query: {query_display}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Initialize state
        initial_state: AgentState = {
            "user_query": query,
            "conversation_id": conversation_id,
            "intent": "",
            "confidence": 0.0,
            "sentiment": "neutral",
            "requires_human": False,
            "classification_reasoning": "",
            "retrieved_documents": [],
            "retrieval_method": "",
            "search_time_ms": 0.0,
            "response": "",
            "current_agent": "",
            "workflow_status": "started",
            "error": None,
            "total_time_ms": 0.0
        }
        
        try:
            # Run workflow
            final_state = await self.app.ainvoke(initial_state)
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            final_state["total_time_ms"] = total_time
            final_state["workflow_status"] = "completed"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ Workflow completed in {total_time:.0f}ms")
            logger.info(f"   Intent: {final_state['intent']}")
            logger.info(f"   Documents: {len(final_state.get('retrieved_documents', []))}")
            logger.info(f"   Status: {final_state['workflow_status']}")
            logger.info(f"{'='*60}\n")
            
            return final_state
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {str(e)}")
            
            total_time = (time.time() - start_time) * 1000
            
            final_state = initial_state.copy()
            final_state["workflow_status"] = "error"
            final_state["error"] = str(e)
            final_state["response"] = "عذراً، حدث خطأ في معالجة استفسارك. يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
            final_state["total_time_ms"] = total_time
            
            return final_state
    
    # ========== SYNCHRONOUS WRAPPER FOR TESTING ==========
    
    def process_query_sync(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Synchronous version of process_query
        
        Args:
            query: User query in Arabic
            conversation_id: Optional conversation ID
            
        Returns:
            Final state with response
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_query(query, conversation_id)
        )