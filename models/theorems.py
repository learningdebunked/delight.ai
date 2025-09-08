"""
Implementation and validation of the three core SEDS theorems:
1. Convergence Theorem
2. Invariance Theorem
3. Fusion Optimality Theorem
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

@dataclass
class ConvergenceMetrics:
    """Metrics for evaluating the Convergence Theorem."""
    loss_history: List[float]
    gradient_norms: List[float]
    convergence_rate: Optional[float] = None
    is_converged: bool = False
    convergence_threshold: float = 1e-6

@dataclass
class InvarianceMetrics:
    """Metrics for evaluating the Invariance Theorem."""
    wasserstein_distances: Dict[str, float]
    feature_importance: Dict[str, float]
    is_invariant: bool = False
    invariance_threshold: float = 0.1

@dataclass
class FusionMetrics:
    """Metrics for evaluating the Fusion Optimality Theorem."""
    modality_weights: Dict[str, float]
    individual_performance: Dict[str, float]
    fused_performance: float
    improvement_ratio: float
    is_optimal: bool = False

class SEDSTheorems:
    """Implementation of the three core SEDS theorems."""
    
    @staticmethod
    def validate_convergence(metrics: ConvergenceMetrics, window: int = 10) -> ConvergenceMetrics:
        """
        Validate the Convergence Theorem.
        
        The theorem states that the system's loss function should converge to a minimum
        at a rate of O(1/t) where t is the number of iterations.
        
        Args:
            metrics: Convergence metrics collected during training
            window: Window size for calculating convergence rate
            
        Returns:
            Updated metrics with convergence analysis
        """
        if len(metrics.loss_history) < 2:
            return metrics
            
        # Calculate gradient norms if not provided
        if not metrics.gradient_norms:
            metrics.gradient_norms = [
                abs(metrics.loss_history[i] - metrics.loss_history[i-1])
                for i in range(1, len(metrics.loss_history))
            ]
        
        # Check if converged
        if len(metrics.gradient_norms) > 0:
            metrics.is_converged = metrics.gradient_norms[-1] < metrics.convergence_threshold
            
            # Calculate convergence rate over the last 'window' steps
            if len(metrics.loss_history) > window:
                recent_losses = metrics.loss_history[-window:]
                t_values = np.arange(len(recent_losses))
                
                # Fit 1/t curve to the recent losses
                def objective(params, t):
                    a, b = params
                    return a / (t + b)
                    
                def loss(params):
                    return np.mean((recent_losses - objective(params, t_values))**2)
                
                result = minimize(
                    loss,
                    x0=[1.0, 1.0],
                    method='L-BFGS-B',
                    bounds=[(0, None), (1e-6, None)]
                )
                
                if result.success:
                    metrics.convergence_rate = result.x[0]
        
        return metrics
    
    @staticmethod
    def validate_invariance(
        source_distributions: Dict[str, np.ndarray],
        target_distributions: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> InvarianceMetrics:
        """
        Validate the Invariance Theorem.
        
        The theorem states that the system should learn representations that are
        invariant to irrelevant transformations of the input.
        
        Args:
            source_distributions: Dictionary of feature distributions from source domain
            target_distributions: Dictionary of feature distributions from target domain
            feature_names: List of feature names
            
        Returns:
            Invariance metrics
        """
        wasserstein_distances = {}
        feature_importance = {}
        
        for feature in feature_names:
            if feature in source_distributions and feature in target_distributions:
                # Calculate Wasserstein distance between source and target distributions
                wd = wasserstein_distance(
                    source_distributions[feature],
                    target_distributions[feature]
                )
                wasserstein_distances[feature] = wd
                
                # Feature importance is inversely related to distribution shift
                # Lower distance = more invariant = higher importance
                feature_importance[feature] = 1.0 / (1.0 + wd)
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        # Check if the system is invariant (all distances below threshold)
        metrics = InvarianceMetrics(
            wasserstein_distances=wasserstein_distances,
            feature_importance=feature_importance,
            is_invariant=all(d < 0.1 for d in wasserstein_distances.values())
        )
        
        return metrics
    
    @staticmethod
    def validate_fusion_optimality(
        modality_performance: Dict[str, float],
        fused_performance: float,
        modality_weights: Dict[str, float]
    ) -> FusionMetrics:
        """
        Validate the Fusion Optimality Theorem.
        
        The theorem states that the fusion of multiple modalities should perform
        at least as well as the best individual modality.
        
        Args:
            modality_performance: Dictionary of performance metrics for each modality
            fused_performance: Performance of the fused model
            modality_weights: Weights used for fusion
            
        Returns:
            Fusion metrics
        """
        best_individual = max(modality_performance.values())
        improvement_ratio = (fused_performance - best_individual) / max(1e-10, best_individual)
        
        metrics = FusionMetrics(
            modality_weights=modality_weights,
            individual_performance=modality_performance,
            fused_performance=fused_performance,
            improvement_ratio=improvement_ratio,
            is_optimal=fused_performance >= best_individual
        )
        
        return metrics
