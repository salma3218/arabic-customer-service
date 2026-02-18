"""
Intent Classifier Agent (Agent 1)
Uses Groq API with Llama-4-Maverick model
"""

import json
import time
from typing import Dict, Any, Optional
from groq import Groq

from .base_agent import BaseAgent
from .prompts import INTENT_CLASSIFICATION_PROMPT
from backend.config.settings import GROQ_API_KEY, INTENT_CATEGORIES
from backend.utils.logger import get_logger, format_query_preview, format_arabic_for_terminal

logger = get_logger(__name__)


class IntentClassifierAgent(BaseAgent):
    """
    Agent 1: Intent Classification
    
    Responsibilities:
    - Classify customer query into one of 6 intent categories
    - Determine confidence score
    - Detect sentiment
    - Decide if human escalation needed
    """
    
    def __init__(self, model_name: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        super().__init__(name="IntentClassifier", model_name=model_name)
        
        # Initialize Groq client
        self.client = Groq(api_key=GROQ_API_KEY)
        
        logger.info(f"✅ IntentClassifierAgent initialized with model: {model_name}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a customer query and classify intent
        
        Args:
            input_data: {
                "query": str,  # Customer query in Arabic
                "conversation_history": Optional[List]  # Not used yet
            }
            
        Returns:
            {
                "intent": str,
                "confidence": float,
                "sentiment": str,
                "requires_human": bool,
                "reasoning": str,
                "inference_time_ms": float,
                "error": Optional[str]
            }
        """
        
        query = input_data.get("query", "")
        
        if not query:
            return self._error_response("Empty query provided")
        
        # ✅ Format Arabic text for terminal display
        query_display = format_query_preview(query, max_length=60)
        logger.info(f"🔍 Classifying: {query_display}")
        
        # Build prompt
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        
        # Call Groq API
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # ✅ CORRECTED: Groq API parameters
                max_tokens=200,      # Not max_new_tokens
                temperature=0        # Deterministic (not None)
            )
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Extract response
            raw_response = response.choices[0].message.content
            
            logger.debug(f"Raw response: {raw_response[:200]}...")
            
            # Parse JSON
            result = self._parse_response(raw_response)
            
            # Validate
            if not self.validate_output(result):
                logger.warning("Invalid output format, using defaults")
                result = self._default_response(query)
            
            # Add metadata
            result["inference_time_ms"] = inference_time
            result["model"] = self.model_name
            
            # ✅ Clean, formatted logging with proper Arabic display
            logger.info(
                f"✅ {result['intent']:20s} | "
                f"conf: {result['confidence']:.2f} | "
                f"sentiment: {result.get('sentiment', 'neutral'):12s} | "
                f"{inference_time:4.0f}ms"
            )
            
            # ✅ Optional: Log reasoning in Arabic if available
            if result.get('reasoning') and logger.level("DEBUG").no <= logger._core.min_level:
                reasoning_display = format_arabic_for_terminal(result['reasoning'], max_length=80)
                logger.debug(f"Reasoning: {reasoning_display}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during classification: {str(e)}")
            return self._error_response(str(e))
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        
        try:
            # Find JSON in response
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = raw_response[json_start:json_end]
            result = json.loads(json_str)
            
            # Normalize intent (handle different formats)
            if 'intent' in result:
                intent = result['intent'].lower().strip()
                intent = intent.replace(' ', '_')
                
                # Validate intent
                if intent not in INTENT_CATEGORIES:
                    logger.warning(f"Invalid intent '{intent}', defaulting to general_question")
                    result['intent'] = "general_question"
                else:
                    result['intent'] = intent
            
            # Ensure all required fields exist with defaults
            result.setdefault('confidence', 0.5)
            result.setdefault('sentiment', 'neutral')
            result.setdefault('requires_human', False)
            result.setdefault('reasoning', 'تصنيف آلي')
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output format"""
        
        required_fields = ['intent', 'confidence', 'sentiment', 'requires_human']
        
        # Check all required fields exist
        if not all(field in output for field in required_fields):
            logger.warning(f"Missing required fields. Got: {list(output.keys())}")
            return False
        
        # Validate intent is in categories
        if output['intent'] not in INTENT_CATEGORIES:
            logger.warning(f"Invalid intent: {output['intent']}")
            return False
        
        # Validate confidence is between 0 and 1
        try:
            conf = float(output['confidence'])
            if not (0 <= conf <= 1):
                logger.warning(f"Confidence out of range: {conf}")
                return False
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence type: {type(output['confidence'])}")
            return False
        
        # Validate sentiment
        valid_sentiments = ['positive', 'neutral', 'negative', 'very_negative']
        if output['sentiment'] not in valid_sentiments:
            logger.warning(f"Invalid sentiment: {output['sentiment']}")
            return False
        
        return True
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Return error response"""
        return {
            "intent": "general_question",
            "confidence": 0.0,
            "sentiment": "neutral",
            "requires_human": True,
            "reasoning": "خطأ في المعالجة",
            "inference_time_ms": 0,
            "error": error_message,
            "model": self.model_name
        }
    
    def _default_response(self, query: str) -> Dict[str, Any]:
        """Return default response when parsing fails"""
        return {
            "intent": "general_question",
            "confidence": 0.1,
            "sentiment": "neutral",
            "requires_human": True,
            "reasoning": "لم يتم التعرف على النية بشكل واضح",
            "model": self.model_name
        }
    
    # ========== SYNCHRONOUS WRAPPER FOR BENCHMARKING ==========
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Synchronous classification (for benchmarking)
        
        Args:
            query: Customer query in Arabic
            
        Returns:
            Classification result dictionary
        """
        import asyncio
        
        # Run async method synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process({"query": query}))