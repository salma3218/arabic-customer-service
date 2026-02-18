"""
Metrics calculation for benchmarking
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    cohen_kappa_score
)
from backend.config.settings import INTENT_CATEGORIES
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive benchmark metrics
    
    Args:
        results: List of result dictionaries from benchmark
        
    Returns:
        Dictionary containing all metrics
    """
    
    logger.info("📊 Calculating metrics...")
    
    # Extract labels
    y_true = [r['true_intent'] for r in results]
    y_pred = [r['predicted_intent'] for r in results]
    
    # 1. MACRO F1-SCORE (Primary Metric)
    macro_f1 = f1_score(
        y_true, y_pred, 
        average='macro', 
        labels=INTENT_CATEGORIES,
        zero_division=0
    )
    
    # 2. WEIGHTED F1-SCORE
    weighted_f1 = f1_score(
        y_true, y_pred,
        average='weighted',
        labels=INTENT_CATEGORIES,
        zero_division=0
    )
    
    # 3. PER-CLASS METRICS
    class_report = classification_report(
        y_true, y_pred,
        target_names=INTENT_CATEGORIES,
        labels=INTENT_CATEGORIES,
        output_dict=True,
        zero_division=0
    )
    
    # 4. COHEN'S KAPPA
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 5. CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred, labels=INTENT_CATEGORIES)
    
    # 6. HUMAN HANDOFF METRICS
    y_true_human = [r['true_requires_human'] for r in results]
    y_pred_human = [r['predicted_requires_human'] for r in results]
    
    if sum(y_pred_human) > 0:
        handoff_precision = precision_score(y_true_human, y_pred_human, zero_division=0)
        handoff_recall = recall_score(y_true_human, y_pred_human, zero_division=0)
        handoff_f1 = f1_score(y_true_human, y_pred_human, zero_division=0)
    else:
        handoff_precision = handoff_recall = handoff_f1 = 0.0
    
    # 7. CALIBRATION ERROR (ECE)
    confidences = np.array([r['confidence'] for r in results])
    correct = np.array([r['true_intent'] == r['predicted_intent'] for r in results])
    ece = calculate_ece(confidences, correct)
    
    # 8. LATENCY METRICS
    latencies = [r['inference_time_ms'] for r in results]
    
    # 9. ERROR RATE
    errors = sum([1 for r in results if r.get('error')])
    error_rate = errors / len(results)
    
    # 10. ACCURACY BY DIFFICULTY
    accuracy_by_difficulty = {}
    for difficulty in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r.get('difficulty') == difficulty]
        if diff_results:
            accuracy = sum([r['true_intent'] == r['predicted_intent'] 
                          for r in diff_results]) / len(diff_results)
            accuracy_by_difficulty[difficulty] = accuracy
    
    metrics = {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'cohen_kappa': kappa,
        'class_report': class_report,
        'confusion_matrix': cm.tolist(),  # Convert to list for JSON
        'handoff_precision': handoff_precision,
        'handoff_recall': handoff_recall,
        'handoff_f1': handoff_f1,
        'ece': ece,
        'latency_p50': np.percentile(latencies, 50),
        'latency_p95': np.percentile(latencies, 95),
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'error_rate': error_rate,
        'accuracy_by_difficulty': accuracy_by_difficulty,
        'total_samples': len(results)
    }
    
    logger.info(f"✅ Metrics calculated: Macro F1 = {macro_f1:.4f}")
    
    return metrics


def calculate_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error
    
    Args:
        confidences: Array of confidence scores
        correct: Array of boolean correctness
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def print_metrics_summary(metrics: Dict[str, Any], model_name: str):
    """Print formatted metrics summary"""
    
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS: {model_name}")
    print(f"{'='*60}\n")
    
    print("🎯 PRIMARY METRICS:")
    print(f"   Macro F1-Score:        {metrics['macro_f1']:.4f} {'✅' if metrics['macro_f1'] > 0.80 else '⚠️'}")
    print(f"   Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    print(f"   Cohen's Kappa:         {metrics['cohen_kappa']:.4f} {'✅' if metrics['cohen_kappa'] > 0.70 else '⚠️'}")
    
    print("\n📝 PER-CLASS F1-SCORES:")
    for category in INTENT_CATEGORIES:
        f1 = metrics['class_report'][category]['f1-score']
        support = metrics['class_report'][category]['support']
        print(f"   {category:20s}: {f1:.4f} (n={int(support)})")
    
    print("\n🔄 HUMAN HANDOFF METRICS:")
    print(f"   Precision:             {metrics['handoff_precision']:.4f}")
    print(f"   Recall:                {metrics['handoff_recall']:.4f} {'✅' if metrics['handoff_recall'] > 0.90 else '⚠️'}")
    print(f"   F1-Score:              {metrics['handoff_f1']:.4f}")
    
    print("\n📊 CALIBRATION:")
    print(f"   ECE (Expected Cal Err): {metrics['ece']:.4f} {'✅' if metrics['ece'] < 0.10 else '⚠️'}")
    
    print("\n⚡ PERFORMANCE:")
    print(f"   Latency (p50):         {metrics['latency_p50']:.0f} ms")
    print(f"   Latency (p95):         {metrics['latency_p95']:.0f} ms")
    print(f"   Latency (mean):        {metrics['latency_mean']:.0f} ms ± {metrics['latency_std']:.0f} ms")
    print(f"   Error Rate:            {metrics['error_rate']*100:.1f}%")
    
    if metrics['accuracy_by_difficulty']:
        print("\n📈 ACCURACY BY DIFFICULTY:")
        for difficulty, accuracy in metrics['accuracy_by_difficulty'].items():
            print(f"   {difficulty.capitalize():10s}: {accuracy:.4f}")
    
    print(f"\n{'='*60}\n")