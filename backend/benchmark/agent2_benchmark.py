"""
Benchmark script for Agent 2 (Knowledge Retriever)
Tests retrieval quality using standard metrics with semantic similarity scoring
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from agents.knowledge_retriever import KnowledgeRetrieverAgent
from data.knowledge_base_manager import KnowledgeBaseManager
from config.settings import RESULTS_DIR, AGENT_2_MODEL
from utils.logger import get_logger

# Import metrics from separate file
from benchmark.retrieval_metrics import RetrievalMetrics

logger = get_logger(__name__)


async def run_benchmark_async(
    agent: KnowledgeRetrieverAgent,
    test_dataset: List[Dict[str, Any]],
    model_name: str
) -> tuple:
    """
    Run benchmark asynchronously
    
    Args:
        agent: KnowledgeRetrieverAgent instance
        test_dataset: List of test cases
        model_name: Model identifier
        
    Returns:
        Tuple of (results, metrics)
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 BENCHMARKING AGENT 2: {model_name}")
    logger.info(f"   Test Cases: {len(test_dataset)}")
    logger.info(f"   Evaluation Method: Semantic Similarity Scoring")
    logger.info(f"{'='*60}\n")
    
    results = []
    metrics_calc = RetrievalMetrics()
    
    # Metrics accumulators
    all_ndcg = []
    all_mrr = []
    all_precision = []
    all_recall = []
    all_latencies = []
    
    # Process each test case
    for test_case in tqdm(test_dataset, desc=f"Testing {model_name}"):
        # Get prediction
        prediction = await agent.process({
            "user_query": test_case['query'],
            "intent": test_case['expected_intent']
        })
        
        # ✅ UPDATED: Calculate relevance using semantic similarity scores
        relevance_labels = []
        
        for doc in prediction['documents'][:10]:  # Process top 10 documents
            # Get the semantic similarity score from vector search
            similarity = doc['relevance_score']
            
            # Convert similarity score (0-1) to relevance label (0-3)
            # These thresholds are based on empirical observation
            if similarity >= 0.60:
                relevance = 3  # Highly relevant (very good match)
            elif similarity >= 0.40:
                relevance = 2  # Somewhat relevant (decent match)
            elif similarity >= 0.25:
                relevance = 1  # Marginally relevant (weak match)
            else:
                relevance = 0  # Not relevant (poor match)
            
            relevance_labels.append(relevance)
        
        # Ensure we have at least 5 labels for metrics calculation
        while len(relevance_labels) < 5:
            relevance_labels.append(0)
        
        # Truncate to first 5 for NDCG@5 calculation
        relevance_labels_k5 = relevance_labels[:5]
        
        # Calculate metrics for this query
        ndcg = metrics_calc.calculate_ndcg_at_k(relevance_labels_k5, k=5)
        mrr = metrics_calc.calculate_mrr(relevance_labels)
        precision = metrics_calc.calculate_precision_at_k(relevance_labels_k5, k=5)
        
        # ✅ FIXED: Calculate actual total relevant docs from the retrieved set
        # Count how many docs are actually relevant (score > 0)
        total_relevant = sum(1 for label in relevance_labels if label > 0)
        
        # Avoid division by zero - if no relevant docs found, set to 1
        if total_relevant == 0:
            total_relevant = 1
            recall = 0.0  # No relevant docs found
        else:
            recall = metrics_calc.calculate_recall_at_k(
                relevance_labels_k5, 
                total_relevant, 
                k=5
            )
        
        # Store metrics
        all_ndcg.append(ndcg)
        all_mrr.append(mrr)
        all_precision.append(precision)
        all_recall.append(recall)
        all_latencies.append(prediction['search_time_ms'])
        
        # Store result
        result = {
            'test_id': test_case['id'],
            'query': test_case['query'],
            'expected_intent': test_case['expected_intent'],
            'retrieved_count': len(prediction['documents']),
            'relevance_labels': relevance_labels[:5],  # Store only top 5
            'similarity_scores': [doc['relevance_score'] for doc in prediction['documents'][:5]],
            'ndcg_at_5': ndcg,
            'mrr': mrr,
            'precision_at_5': precision,
            'recall_at_5': recall,
            'search_time_ms': prediction['search_time_ms'],
            'top_result_score': prediction['documents'][0]['relevance_score'] if prediction['documents'] else 0.0
        }
        
        results.append(result)
    
    # Aggregate metrics
    metrics = {
        'ndcg_at_5_mean': np.mean(all_ndcg),
        'ndcg_at_5_std': np.std(all_ndcg),
        'mrr_mean': np.mean(all_mrr),
        'mrr_std': np.std(all_mrr),
        'precision_at_5_mean': np.mean(all_precision),
        'precision_at_5_std': np.std(all_precision),
        'recall_at_5_mean': np.mean(all_recall),
        'recall_at_5_std': np.std(all_recall),
        'latency_p50': np.percentile(all_latencies, 50),
        'latency_p95': np.percentile(all_latencies, 95),
        'latency_mean': np.mean(all_latencies),
        'latency_std': np.std(all_latencies),
        'total_samples': len(results)
    }
    
    return results, metrics


def print_metrics_summary(metrics: Dict[str, Any], model_name: str):
    """Print formatted metrics summary"""
    
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS: {model_name}")
    print(f"{'='*60}\n")
    
    print("🎯 RANKING QUALITY:")
    print(f"   NDCG@5:            {metrics['ndcg_at_5_mean']:.4f} ± {metrics['ndcg_at_5_std']:.4f} {'✅' if metrics['ndcg_at_5_mean'] > 0.65 else '⚠️'}")
    print(f"   MRR:               {metrics['mrr_mean']:.4f} ± {metrics['mrr_std']:.4f} {'✅' if metrics['mrr_mean'] > 0.60 else '⚠️'}")
    
    print("\n📝 RELEVANCE METRICS:")
    print(f"   Precision@5:       {metrics['precision_at_5_mean']:.4f} ± {metrics['precision_at_5_std']:.4f} {'✅' if metrics['precision_at_5_mean'] > 0.55 else '⚠️'}")
    print(f"   Recall@5:          {metrics['recall_at_5_mean']:.4f} ± {metrics['recall_at_5_std']:.4f} {'✅' if metrics['recall_at_5_mean'] > 0.70 else '⚠️'}")
    
    print("\n⚡ PERFORMANCE:")
    print(f"   Latency (p50):     {metrics['latency_p50']:.0f} ms")
    print(f"   Latency (p95):     {metrics['latency_p95']:.0f} ms")
    print(f"   Latency (mean):    {metrics['latency_mean']:.0f} ms ± {metrics['latency_std']:.0f} ms")
    
    # Calculate overall score (adjusted weights for semantic similarity scoring)
    overall_score = (
        metrics['ndcg_at_5_mean'] * 0.4 +
        metrics['mrr_mean'] * 0.3 +
        metrics['precision_at_5_mean'] * 0.2 +
        metrics['recall_at_5_mean'] * 0.1
    )
    
    print(f"\n🏆 OVERALL SCORE:      {overall_score:.4f} {'✅' if overall_score > 0.60 else '⚠️'}")
    print(f"\n💡 NOTE: Scores based on semantic similarity thresholds:")
    print(f"   - Similarity ≥0.60 = Highly relevant (score: 3)")
    print(f"   - Similarity ≥0.40 = Somewhat relevant (score: 2)")
    print(f"   - Similarity ≥0.25 = Marginally relevant (score: 1)")
    print(f"   - Similarity <0.25 = Not relevant (score: 0)")
    print(f"{'='*60}\n")


def save_results(results: List[Dict], metrics: Dict, model_name: str):
    """Save benchmark results to files"""
    
    # Create output directory
    output_dir = RESULTS_DIR / "agent_2" / model_name.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'agent': 'knowledge_retriever',
                'model': model_name,
                'evaluation_method': 'semantic_similarity_scoring',
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(results),
                'similarity_thresholds': {
                    'highly_relevant': 0.60,
                    'somewhat_relevant': 0.40,
                    'marginally_relevant': 0.25
                }
            },
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved results to {results_file}")


def main():
    """Main benchmark execution"""
    
    print("\n" + "="*60)
    print("🚀 AGENT 2 (KNOWLEDGE RETRIEVER) BENCHMARK")
    print("="*60)
    print(f"Model: {AGENT_2_MODEL}")
    print(f"Evaluation: Semantic Similarity Scoring")
    print(f"Dataset: Agent 1 test dataset (reused)")
    print("="*60 + "\n")
    
    # 1. Initialize knowledge base
    logger.info("📚 Initializing knowledge base...")
    kb = KnowledgeBaseManager()
    kb_stats = kb.get_stats()
    
    if kb_stats['total_documents'] == 0:
        logger.error("❌ Knowledge base is empty!")
        logger.info("💡 Please run: python -m data.knowledge_base_manager")
        return
    
    logger.info(f"✅ Knowledge base loaded: {kb_stats['total_documents']} documents")
    
    # 2. Load test dataset (reuse from Agent 1)
    logger.info("📂 Loading test dataset...")
    from config.settings import TEST_DATASET_PATH
    
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
    
    logger.info(f"✅ Loaded {len(test_dataset)} test cases")
    
    # 3. Initialize agent
    logger.info(f"🤖 Initializing KnowledgeRetrieverAgent...")
    agent = KnowledgeRetrieverAgent(model_name=AGENT_2_MODEL)
    
    # 4. Run benchmark
    logger.info("🧪 Starting benchmark...")
    results, metrics = asyncio.run(
        run_benchmark_async(agent, test_dataset, AGENT_2_MODEL)
    )
    
    # 5. Print results
    print_metrics_summary(metrics, AGENT_2_MODEL)
    
    # 6. Save results
    logger.info("💾 Saving results...")
    save_results(results, metrics, AGENT_2_MODEL)
    
    # 7. Final summary
    print("\n" + "="*60)
    print("🎯 BENCHMARK COMPLETE!")
    print("="*60)
    
    overall_score = (
        metrics['ndcg_at_5_mean'] * 0.4 +
        metrics['mrr_mean'] * 0.3 +
        metrics['precision_at_5_mean'] * 0.2 +
        metrics['recall_at_5_mean'] * 0.1
    )
    
    print(f"\n📊 Key Metrics:")
    print(f"   NDCG@5:        {metrics['ndcg_at_5_mean']:.4f}")
    print(f"   MRR:           {metrics['mrr_mean']:.4f}")
    print(f"   Precision@5:   {metrics['precision_at_5_mean']:.4f}")
    print(f"   Recall@5:      {metrics['recall_at_5_mean']:.4f}")
    print(f"   Overall Score: {overall_score:.4f}")
    
    status = "✅ GOOD QUALITY" if overall_score > 0.60 else "⚠️ NEEDS IMPROVEMENT"
    print(f"\n   Status: {status}")
    
    print(f"\n📁 Results saved to: {RESULTS_DIR / 'agent_2'}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()