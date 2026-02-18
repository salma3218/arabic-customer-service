"""
Retrieval Metrics for Agent 2 Benchmarking

Provides standard information retrieval metrics:
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
"""

from typing import List, Dict, Any
import numpy as np


class RetrievalMetrics:
    """Calculate retrieval quality metrics"""
    
    @staticmethod
    def calculate_ndcg_at_k(relevance_labels: List[int], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain
        Measures ranking quality
        
        Args:
            relevance_labels: [3, 2, 1, 0, 0] (3=highly relevant, 0=irrelevant)
            k: Number of top results to consider
            
        Returns:
            NDCG score (0-1, higher is better)
            
        Example:
            >>> labels = [3, 2, 1, 0, 0]  # Perfect ranking
            >>> RetrievalMetrics.calculate_ndcg_at_k(labels, k=5)
            1.0
            
            >>> labels = [0, 0, 3, 2, 1]  # Poor ranking
            >>> RetrievalMetrics.calculate_ndcg_at_k(labels, k=5)
            0.45
        """
        if not relevance_labels:
            return 0.0
        
        # Ensure we have exactly k scores
        labels = relevance_labels[:k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, rel in enumerate(labels):
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Calculate IDCG (Ideal DCG)
        ideal_labels = sorted(labels, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_labels):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def calculate_mrr(relevance_labels: List[int]) -> float:
        """
        Mean Reciprocal Rank
        Measures position of first relevant result
        
        Args:
            relevance_labels: [0, 0, 3, 2, 1] (first relevant at position 3)
            
        Returns:
            MRR score (0-1, higher is better)
            
        Example:
            >>> labels = [3, 2, 1, 0, 0]  # First relevant at position 1
            >>> RetrievalMetrics.calculate_mrr(labels)
            1.0
            
            >>> labels = [0, 0, 3, 2, 1]  # First relevant at position 3
            >>> RetrievalMetrics.calculate_mrr(labels)
            0.333
        """
        for i, rel in enumerate(relevance_labels):
            if rel > 0:  # First relevant result
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def calculate_precision_at_k(relevance_labels: List[int], k: int = 5) -> float:
        """
        Precision@K
        What % of top K results are relevant?
        
        Args:
            relevance_labels: [3, 2, 0, 1, 0] (3 relevant in top 5)
            k: Number of top results to consider
            
        Returns:
            Precision (0-1, higher is better)
            
        Example:
            >>> labels = [3, 2, 0, 1, 0]  # 3 relevant out of 5
            >>> RetrievalMetrics.calculate_precision_at_k(labels, k=5)
            0.6
        """
        if not relevance_labels:
            return 0.0
        
        top_k = relevance_labels[:k]
        relevant = sum(1 for rel in top_k if rel > 0)
        
        return relevant / len(top_k)
    
    @staticmethod
    def calculate_recall_at_k(relevance_labels: List[int], total_relevant: int, k: int = 5) -> float:
        """
        Recall@K
        What % of all relevant docs did we retrieve in top K?
        
        Args:
            relevance_labels: [3, 2, 0, 0, 0] (2 retrieved)
            total_relevant: 5 (5 total relevant docs exist)
            k: Number of top results to consider
            
        Returns:
            Recall (0-1, higher is better)
            
        Example:
            >>> labels = [3, 2, 0, 0, 0]  # 2 relevant retrieved
            >>> RetrievalMetrics.calculate_recall_at_k(labels, total_relevant=5, k=5)
            0.4  # Retrieved 2 out of 5 relevant
        """
        if total_relevant == 0:
            return 1.0
        
        top_k = relevance_labels[:k]
        found = sum(1 for rel in top_k if rel > 0)
        
        return found / total_relevant
    
    @staticmethod
    def calculate_map(all_relevance_labels: List[List[int]]) -> float:
        """
        Mean Average Precision
        Average precision across multiple queries
        
        Args:
            all_relevance_labels: [[3,2,0], [0,3,2], ...] for multiple queries
            
        Returns:
            MAP score (0-1, higher is better)
        """
        if not all_relevance_labels:
            return 0.0
        
        avg_precisions = []
        
        for relevance_labels in all_relevance_labels:
            # Calculate average precision for this query
            precisions = []
            num_relevant = 0
            
            for i, rel in enumerate(relevance_labels):
                if rel > 0:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    precisions.append(precision_at_i)
            
            if precisions:
                avg_precision = sum(precisions) / len(precisions)
                avg_precisions.append(avg_precision)
        
        if avg_precisions:
            return sum(avg_precisions) / len(avg_precisions)
        
        return 0.0


# Convenience function for calculating all metrics at once
def calculate_all_metrics(
    relevance_labels: List[int],
    total_relevant: int = None,
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics for a single query
    
    Args:
        relevance_labels: Relevance scores for retrieved documents
        total_relevant: Total number of relevant documents (for recall)
        k: Number of top results to consider
        
    Returns:
        Dictionary with all metrics
        
    Example:
        >>> labels = [3, 2, 1, 0, 0]
        >>> metrics = calculate_all_metrics(labels, total_relevant=5, k=5)
        >>> print(metrics)
        {
            'ndcg_at_5': 0.95,
            'mrr': 1.0,
            'precision_at_5': 0.6,
            'recall_at_5': 0.6
        }
    """
    
    metrics_calc = RetrievalMetrics()
    
    metrics = {
        f'ndcg_at_{k}': metrics_calc.calculate_ndcg_at_k(relevance_labels, k),
        'mrr': metrics_calc.calculate_mrr(relevance_labels),
        f'precision_at_{k}': metrics_calc.calculate_precision_at_k(relevance_labels, k),
    }
    
    if total_relevant is not None:
        metrics[f'recall_at_{k}'] = metrics_calc.calculate_recall_at_k(
            relevance_labels, 
            total_relevant, 
            k
        )
    
    return metrics