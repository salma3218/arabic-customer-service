"""
Visualization functions for benchmark results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any
from backend.config.settings import INTENT_CATEGORIES, RESULTS_DIR
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path = None):
    """Plot confusion matrix heatmap"""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=INTENT_CATEGORIES,
        yticklabels=INTENT_CATEGORIES,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Intent', fontsize=12)
    plt.xlabel('Predicted Intent', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_per_class_f1(class_report: Dict, model_name: str, save_path: Path = None):
    """Plot F1 scores by class"""
    
    categories = INTENT_CATEGORIES
    f1_scores = [class_report[cat]['f1-score'] for cat in categories]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(categories)), f1_scores, color='steelblue', edgecolor='navy')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Intent Category', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'Per-Class F1-Scores - {model_name}', fontsize=16, fontweight='bold')
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='Target (0.8)')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved F1 chart to {save_path}")
    
    plt.close()


def plot_calibration_curve(confidences: np.ndarray, correct: np.ndarray, 
                          model_name: str, save_path: Path = None):
    """Plot calibration curve"""
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(correct[in_bin].mean())
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    plt.plot(bin_centers, bin_accuracies, 'o-', color='steelblue', 
            label='Model Calibration', linewidth=2, markersize=8)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Calibration Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved calibration curve to {save_path}")
    
    plt.close()


def save_all_visualizations(metrics: Dict[str, Any], results: list, 
                           model_name: str):
    """Generate and save all visualizations"""
    
    logger.info("🎨 Generating visualizations...")
    
    # Create output directory
    output_dir = RESULTS_DIR / model_name.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm, 
        model_name,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # 2. Per-Class F1
    plot_per_class_f1(
        metrics['class_report'],
        model_name,
        save_path=output_dir / 'f1_by_class.png'
    )
    
    # 3. Calibration Curve
    confidences = np.array([r['confidence'] for r in results])
    correct = np.array([r['true_intent'] == r['predicted_intent'] for r in results])
    plot_calibration_curve(
        confidences,
        correct,
        model_name,
        save_path=output_dir / 'calibration_curve.png'
    )
    
    logger.info(f"✅ All visualizations saved to {output_dir}")