"""
Test script for LangGraph Multi-Agent Workflow
Tests Agents 1 + 2 orchestration
"""

import asyncio
from pathlib import Path
from backend.orchestrator.workflow import MultiAgentWorkflow
from backend.data.knowledge_base_manager import KnowledgeBaseManager
from backend.utils.logger import get_logger, format_arabic_for_terminal

logger = get_logger(__name__)


async def test_single_query(workflow: MultiAgentWorkflow, query: str):
    """Test a single query through the workflow"""
    
    print(f"\n{'='*70}")
    print(f"🧪 TEST QUERY: {format_arabic_for_terminal(query, 60)}")
    print(f"{'='*70}\n")
    
    # Process query
    result = await workflow.process_query(query)
    
    # Display results
    print(f"📊 WORKFLOW RESULTS:")
    print(f"   Intent:          {result['intent']}")
    print(f"   Confidence:      {result['confidence']:.2f}")
    print(f"   Sentiment:       {result['sentiment']}")
    print(f"   Requires Human:  {result['requires_human']}")
    print(f"   Status:          {result['workflow_status']}")
    
    if result['retrieved_documents']:
        print(f"\n📚 RETRIEVED DOCUMENTS: {len(result['retrieved_documents'])}")
        for i, doc in enumerate(result['retrieved_documents'][:3], 1):
            print(f"\n   [{i}] Score: {doc['relevance_score']:.2f}")
            print(f"       Category: {doc['category']}")
            question = format_arabic_for_terminal(doc['question'], 60)
            print(f"       Question: {question}")
    
    print(f"\n💬 RESPONSE:")
    response = format_arabic_for_terminal(result['response'], 150)
    print(f"   {response}")
    
    print(f"\n⏱️  TIMING:")
    print(f"   Total Time:      {result['total_time_ms']:.0f} ms")
    print(f"   Search Time:     {result.get('search_time_ms', 0):.0f} ms")
    
    print(f"\n{'='*70}\n")
    
    return result


async def run_test_suite():
    """Run comprehensive test suite"""
    
    print("\n" + "="*70)
    print("🚀 LANGGRAPH MULTI-AGENT WORKFLOW TEST SUITE")
    print("="*70 + "\n")
    
    # 1. Check knowledge base
    logger.info("📚 Checking knowledge base...")
    kb = KnowledgeBaseManager()
    kb_stats = kb.get_stats()
    
    if kb_stats['total_documents'] == 0:
        logger.error("❌ Knowledge base is empty!")
        logger.info("💡 Please run: python -m data.knowledge_base_manager")
        return
    
    logger.info(f"✅ Knowledge base ready: {kb_stats['total_documents']} documents")
    
    # 2. Initialize workflow
    logger.info("🔧 Initializing workflow...")
    workflow = MultiAgentWorkflow()
    
    # 3. Test cases
    test_cases = [
        {
            "name": "Product Inquiry (Normal Flow)",
            "query": "ما هي أسعار الباقات المتاحة؟",
            "expected_intent": "product_inquiry",
            "expected_routing": "retrieve"
        },
        {
            "name": "Technical Support (Normal Flow)",
            "query": "كيف أعيد تعيين كلمة المرور؟",
            "expected_intent": "technical_support",
            "expected_routing": "retrieve"
        },
        {
            "name": "Billing Question (Normal Flow)",
            "query": "ما هي طرق الدفع المتاحة؟",
            "expected_intent": "billing_question",
            "expected_routing": "retrieve"
        },
        {
            "name": "Complaint (Human Handoff)",
            "query": "أنا غاضب جداً من الخدمة السيئة والدعم الفني لا يرد!",
            "expected_intent": "complaint",
            "expected_routing": "human"
        },
        {
            "name": "General Question (Normal Flow)",
            "query": "ما هي ساعات العمل لديكم؟",
            "expected_intent": "general_question",
            "expected_routing": "retrieve"
        }
    ]
    
    # 4. Run tests
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'#'*70}")
        
        result = await test_single_query(workflow, test_case['query'])
        
        # Validate result
        success = True
        
        # Check intent
        if result['intent'] != test_case['expected_intent']:
            logger.warning(f"⚠️  Intent mismatch: expected {test_case['expected_intent']}, got {result['intent']}")
            success = False
        
        # Check routing
        if test_case['expected_routing'] == 'retrieve':
            if result['workflow_status'] not in ['documents_retrieved', 'completed']:
                logger.warning(f"⚠️  Expected documents, but got: {result['workflow_status']}")
                success = False
        elif test_case['expected_routing'] == 'human':
            if result['workflow_status'] != 'escalated_to_human':
                logger.warning(f"⚠️  Expected human handoff, but got: {result['workflow_status']}")
                success = False
        
        if success:
            print("✅ TEST PASSED")
            passed += 1
        else:
            print("❌ TEST FAILED")
            failed += 1
        
        results.append({
            'test_name': test_case['name'],
            'success': success,
            'result': result
        })
    
    # 5. Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"   Total Tests:  {len(test_cases)}")
    print(f"   Passed:       {passed} ✅")
    print(f"   Failed:       {failed} ❌")
    print(f"   Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main test execution"""
    
    # Run async test suite
    asyncio.run(run_test_suite())


if __name__ == "__main__":
    main()
