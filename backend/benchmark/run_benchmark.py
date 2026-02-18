"""
Main benchmark script for Intent Classifier Agent
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

from backend.agents.intent_classifier import IntentClassifierAgent
from backend.benchmark.metrics import calculate_metrics, print_metrics_summary
from backend.benchmark.visualizations import save_all_visualizations
from backend.config.settings import TEST_DATASET_PATH, RESULTS_DIR, PRIMARY_MODEL
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def run_benchmark_async(
    agent: IntentClassifierAgent,
    test_dataset: List[Dict[str, Any]],
    model_name: str
) -> tuple:
    """
    Run benchmark asynchronously
    
    Args:
        agent: IntentClassifierAgent instance
        test_dataset: List of test cases
        model_name: Model identifier
        
    Returns:
        Tuple of (results, metrics)
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 BENCHMARKING: {model_name}")
    logger.info(f"   Test Cases: {len(test_dataset)}")
    logger.info(f"   Mode: DETERMINISTIC (do_sample=False)")
    logger.info(f"{'='*60}\n")
    
    results = []
    
    # Process each test case
    for test_case in tqdm(test_dataset, desc=f"Testing {model_name}"):
        # Get prediction
        prediction = await agent.process({"query": test_case['query']})
        
        # Store result
        result = {
            'test_id': test_case['id'],
            'query': test_case['query'],
            'true_intent': test_case['expected_intent'],
            'predicted_intent': prediction['intent'],
            'true_requires_human': test_case['requires_human'],
            'predicted_requires_human': prediction.get('requires_human', False),
            'confidence': prediction.get('confidence', 0.0),
            'sentiment': prediction.get('sentiment', 'neutral'),
            'difficulty': test_case.get('difficulty', 'medium'),
            'inference_time_ms': prediction.get('inference_time_ms', 0),
            'error': prediction.get('error'),
            'reasoning': prediction.get('reasoning', '')
        }
        
        results.append(result)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    return results, metrics


def save_results(results: List[Dict], metrics: Dict, model_name: str):
    """Save benchmark results to files"""
    
    # Create output directory
    output_dir = RESULTS_DIR / model_name.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(results),
                'generation_mode': 'deterministic'
            },
            'metrics': {k: (v if not isinstance(v, (list, dict)) else None) 
                       for k, v in metrics.items()},
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved results to {results_file}")
    
    # Save metrics summary
    metrics_file = output_dir / 'metrics_summary.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        # Filter out numpy arrays for JSON serialization
        json_metrics = {
            k: (float(v) if isinstance(v, (float, int)) else 
                (v if k != 'confusion_matrix' else None))
            for k, v in metrics.items()
        }
        json.dump(json_metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved metrics to {metrics_file}")


def main():
    """Main benchmark execution"""
    
    print("\n" + "="*60)
    print("🚀 ARABIC INTENT CLASSIFICATION BENCHMARK")
    print("="*60)
    print(f"Model: {PRIMARY_MODEL}")
    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Mode: DETERMINISTIC (Reproducible)")
    print("="*60 + "\n")
    
    # 1. Load test dataset
    logger.info(f"📂 Loading test dataset from {TEST_DATASET_PATH}...")
    
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
    
    logger.info(f"✅ Loaded {len(test_dataset)} test cases")
    
    # 2. Initialize agent
    logger.info(f"🤖 Initializing IntentClassifierAgent...")
    agent = IntentClassifierAgent(model_name=PRIMARY_MODEL)
    
    # 3. Run benchmark
    logger.info("🧪 Starting benchmark...")
    results, metrics = asyncio.run(
        run_benchmark_async(agent, test_dataset, PRIMARY_MODEL)
    )
    
    # 4. Print results
    print_metrics_summary(metrics, PRIMARY_MODEL)
    
    # 5. Save results
    logger.info("💾 Saving results...")
    save_results(results, metrics, PRIMARY_MODEL)
    
    # 6. Generate visualizations
    save_all_visualizations(metrics, results, PRIMARY_MODEL)
    
    # 7. Final summary
    print("\n" + "="*60)
    print("🎯 BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\n📊 Key Metrics:")
    print(f"   Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"   Cohen's Kappa:  {metrics['cohen_kappa']:.4f}")
    print(f"   ECE:            {metrics['ece']:.4f}")
    print(f"   Latency (p95):  {metrics['latency_p95']:.0f} ms")
    
    status = "✅ PRODUCTION READY" if metrics['macro_f1'] > 0.85 else "⚠️ NEEDS IMPROVEMENT"
    print(f"\n   Status: {status}")
    
    print(f"\n📁 Results saved to: {RESULTS_DIR / PRIMARY_MODEL.replace('/', '_')}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()