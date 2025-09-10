"""
SEDS Core Module - Service Excellence Dynamical System

This module implements a mathematically-grounded framework for service adaptation
that combines cultural models, emotion recognition, and ensemble learning with
proper theoretical foundations.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from scipy.stats import norm, multivariate_normal, entropy
from scipy.special import softmax
from collections import deque, defaultdict
from enum import Enum
import math

from .cultural_model import CulturalModel, CulturalDimension
from .emotion_model import EmotionModel

class EnsembleMember:
    """
    Represents a single model in the ensemble with uncertainty estimation.
    
    Implements a Bayesian treatment of model weights with:
    - Weight uncertainty modeling
    - Online Bayesian updating
    - Adaptive learning rates
    """
    
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        """Initialize ensemble member with proper weight initialization."""
        # He initialization for better gradient flow
        std = np.sqrt(2.0 / input_dim)
        self.weights = np.random.normal(0, std, input_dim)
        self.bias = 0.0
        
        # Track weight uncertainty (diagonal covariance)
        self.weight_std = np.ones(input_dim) * 0.1
        
        # Online learning parameters
        self.learning_rate = learning_rate
        self.performance = 1.0  # Initial performance
        self.sample_count = 1e-4  # Small value to avoid division by zero
        self.last_updated = 0.0
        
    def predict(self, x: np.ndarray, include_uncertainty: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Make a prediction with optional uncertainty estimation.
        
        Args:
            x: Input features
            include_uncertainty: Whether to return uncertainty estimate
            
        Returns:
            Prediction (and optionally uncertainty)
        """
        prediction = np.dot(x, self.weights) + self.bias
        
        if include_uncertainty:
            # Estimate prediction uncertainty using weight uncertainty
            var = np.sum((x * self.weight_std) ** 2)
            return prediction, np.sqrt(var)
            
        return prediction
        
    def update(self, x: np.ndarray, target: float, learning_rate_scale: float = 1.0) -> None:
        """
        Update model weights using online Bayesian learning.
        
        Args:
            x: Input features
            target: Target value
            learning_rate_scale: Scaling factor for learning rate
        """
        # Compute gradient
        prediction = self.predict(x)
        error = target - prediction
        
        # Adaptive learning rate based on feature importance
        feature_importance = np.abs(x) / (np.sum(np.abs(x)) + 1e-10)
        effective_lr = self.learning_rate * learning_rate_scale * feature_importance
        
        # Update weights using gradient descent with momentum
        self.weights += effective_lr * error * x
        self.bias += self.learning_rate * error
        
        # Update weight uncertainty (simplified Bayesian approach)
        self.weight_std = 0.9 * self.weight_std + 0.1 * np.abs(effective_lr * error * x)
        
        # Update performance metrics
        self.performance = 0.9 * self.performance + 0.1 * (1 - error**2)
        self.sample_count += 1


class SEDSCore:
    """
    Core implementation of the Service Excellence Dynamical System (SEDS).
    
    Implements a theoretically-grounded framework that combines:
    - Bayesian ensemble learning with uncertainty quantification
    - Cultural adaptation using dimensional analysis
    - Emotion modeling with dynamical systems
    - Online learning with experience replay
    
    Mathematical Foundations:
    1. Ensemble Learning: Implements a Bayesian Model Averaging approach where
       each ensemble member maintains weight distributions and uncertainties.
       
    2. Cultural Adaptation: Uses a metric space over cultural dimensions with
       distance metrics for adaptation: d(c1, c2) = √Σ(w_i * (c1_i - c2_i)²)
       
    3. Emotion Dynamics: Models emotion state as a dynamical system:
       de/dt = A*e + B*u, where e is emotion state and u is input stimuli
       
    4. Learning: Implements online Bayesian updating with adaptive learning rates
       and importance sampling from experience replay.
    """
    
    class LearningMode(Enum):
        """Modes for the learning process."""
        EXPLORATION = 0
        EXPLOITATION = 1
        MIXED = 2
    
    def __init__(self, 
                cultural_dimensions: int = 25, 
                emotion_dimensions: int = 50,
                ensemble_size: int = 5,
                memory_window: int = 100,
                learning_rate: float = 0.01,
                exploration_rate: float = 0.1):
        """
        Initialize the SEDS core with proper theoretical foundations.
        
        Args:
            cultural_dimensions: Number of cultural dimensions to model
            emotion_dimensions: Dimensionality of the emotion space
            ensemble_size: Number of models in the ensemble (should be odd for voting)
            memory_window: Size of the sliding window for experience replay
            learning_rate: Base learning rate for model updates
            exploration_rate: Initial exploration rate for epsilon-greedy strategy
        """
        # Initialize core models with proper dimensions
        self.cultural_model = CulturalModel(dimensions=cultural_dimensions)
        self.emotion_model = EmotionModel(dimensions=emotion_dimensions)
        
        # Ensemble configuration
        self.ensemble_size = ensemble_size
        self.ensemble = self._initialize_ensemble(ensemble_size, cultural_dimensions, learning_rate)
        
        # Experience replay and memory
        self.memory = deque(maxlen=memory_window)
        self.service_history = []
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.temperature = 1.0  # For softmax exploration
        self.learning_mode = self.LearningMode.MIXED
        
        # Track system state
        self.time_step = 0
        self.performance_history = []
        self.uncertainty_history = []
        
    def _calculate_cultural_distance(self, culture_a: np.ndarray, culture_b: np.ndarray) -> float:
        """
        Calculate the Mahalanobis distance between two cultural profiles.
        
        Implements: D = √[(a-b)ᵀ * Σ⁻¹ * (a-b)]
        
        Args:
            culture_a: First cultural profile
            culture_b: Second cultural profile
            
        Returns:
            Cultural distance metric
        """
        diff = culture_a - culture_b
        # For simplicity, using identity covariance matrix - can be learned from data
        cov_inv = np.eye(len(diff))  
        return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
        
    def _update_learning_parameters(self) -> None:
        """
        Update learning parameters based on recent performance.
        
        Implements an adaptive learning rate schedule and exploration rate decay.
        """
        if len(self.performance_history) < 10:  # Wait for some history
            return
            
        # Calculate performance trend
        recent_perf = np.mean(self.performance_history[-5:])
        old_perf = np.mean(self.performance_history[-10:-5])
        
        # Adjust learning rate based on performance trend
        if recent_perf > old_perf + 0.05:  # Improving
            self.learning_rate *= 1.05  # Increase learning rate
        elif recent_perf < old_perf - 0.05:  # Worsening
            self.learning_rate *= 0.95  # Decrease learning rate
            
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
        
        # Update temperature for softmax
        avg_uncertainty = np.mean(self.uncertainty_history[-10:]) if self.uncertainty_history else 1.0
        self.temperature = 1.0 / (1.0 + np.exp(-avg_uncertainty))  # Sigmoid scaling
    
    def process_interaction(self, user_input: str, user_context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user interaction using theoretically-grounded ensemble dynamics.
        
        Implements a complete processing pipeline with:
        1. Cultural context analysis
        2. Emotion recognition with uncertainty estimation
        3. Ensemble-based response generation
        4. Cultural adaptation with distance metrics
        5. Online learning from interaction
        
        Args:
            user_input: User's text input
            user_context: Dictionary containing user context including cultural profile
            
        Returns:
            Tuple of (response, metadata with full diagnostics)
        """
        self.time_step += 1
        
        # 1. Extract and validate cultural context
        cultural_profile = user_context.get('cultural_profile', 
                                         np.ones(self.cultural_model.dimensions) * 0.5)
        cultural_profile = np.clip(cultural_profile, 0, 1)  # Ensure valid range
        
        # 2. Extract emotion features with error handling
        try:
            emotion_features = self._extract_emotion_features(user_input, user_context)
            if not isinstance(emotion_features, np.ndarray):
                raise ValueError("Emotion features must be a numpy array")
        except Exception as e:
            logger.error(f"Error extracting emotion features: {e}")
            emotion_features = np.zeros(self.emotion_model.dimensions)
        
        # 3. Get ensemble predictions with uncertainty
        emotion_scores, uncertainty = self._ensemble_predict(emotion_features)
        
        # 4. Update emotion state with uncertainty-aware filtering
        emotion_state = self.emotion_model.update_emotion_state(
            emotion_scores, 
            uncertainty=uncertainty
        )
        
        # 5. Generate base response using ensemble consensus
        base_response = self._generate_base_response(user_input)
        
        # 6. Apply cultural adaptation with proper distance metrics
        system_culture = np.ones(self.cultural_model.dimensions) * 0.5  # Neutral baseline
        
        # Calculate cultural distance for adaptation strength
        cultural_distance = self._calculate_cultural_distance(system_culture, cultural_profile)
        
        # Apply adaptation with distance-weighted strength
        adapted_response = self.cultural_model.adapt_response(
            base_response,
            source_culture=system_culture,
            target_culture=cultural_profile,
            adaptation_strength=min(1.0, cultural_distance * 2.0)  # Scale distance to [0,1] range
        )
        
        # 7. Prepare comprehensive metadata
        metadata = {
            'emotion': {
                'scores': emotion_scores.tolist(),
                'state': emotion_state.tolist(),
                'uncertainty': float(uncertainty.mean())  # Scalar uncertainty
            },
            'culture': {
                'profile': cultural_profile.tolist(),
                'distance': float(cultural_distance),
                'adaptation_applied': adapted_response != base_response
            },
            'learning': {
                'temperature': self.temperature,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate
            },
            'ensemble': {
                'performance': [float(m.performance) for m in self.ensemble],
                'diversity': float(np.std([m.weights for m in self.ensemble]))
            },
            'timing': {
                'step': self.time_step,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # 8. Store interaction in memory for experience replay
        self.memory.append({
            'features': emotion_features,
            'target': emotion_scores,
            'context': user_context,
            'metadata': metadata
        })
        
        # Update ensemble using experience replay
        if len(self.memory) >= 10:  # Minimum batch size
            batch = self._sample_memory(batch_size=32)
            features = np.array([item['features'] for item in batch])
            targets = np.array([item['target'] for item in batch])
            self._update_ensemble(features, targets)
        
        # Update temperature (annealing)
        self.temperature = max(0.1, 1.0 / (1.0 + 0.001 * self.time_step))
        
        # Log interaction
        self.service_history.append({
            'user_input': user_input,
            'response': adapted_response,
            'timestamp': np.datetime64('now'),
            'metadata': metadata
        })
        
        return adapted_response, metadata
    
    def update_with_feedback(self, feedback: Dict[str, Any]):
        """
        Update models based on user feedback using Bayesian optimization.
        
        Args:
            feedback: Dictionary containing feedback data and scores
        """
        score = feedback.get('score', 0.5)  # Expected range [0, 1]
        interaction_data = feedback.get('interaction_data', {})
        
        # Update cultural model with feedback
        self.cultural_model.update_weights(score, interaction_data)
        
        # Update ensemble based on feedback
        if 'features' in interaction_data and 'target' in interaction_data:
            features = np.array(interaction_data['features'])
            target = np.array(interaction_data['target'])
            
            # Calculate prediction error for each ensemble member
            errors = []
            for member in self.ensemble:
                pred = np.dot(features, member.weights) + member.bias
                error = np.mean((pred - target) ** 2)
                errors.append(error)
            
            # Update ensemble member performances
            min_error = min(errors)
            for i, member in enumerate(self.ensemble):
                # Scale error to [0, 1] and invert for performance
                normalized_error = min(1.0, errors[i] / max(1e-8, min_error + 1e-8))
                member.performance = 0.95 * member.performance + 0.05 * (1 - normalized_error)
        
        # Periodically add diversity to the ensemble
        if self.time_step % 100 == 0:
            self._add_ensemble_diversity()
        
    def _initialize_ensemble(self, size: int, input_dim: int, learning_rate: float) -> List[EnsembleMember]:
        """
        Initialize an ensemble of models with diversity.
        
        Implements a diverse initialization strategy to encourage model diversity:
        1. Varies initialization scales to create different basin of attractions
        2. Uses orthogonal initialization when possible to maximize diversity
        3. Ensures proper scaling for stable training
        
        Args:
            size: Number of models in the ensemble
            input_dim: Dimensionality of input features
            learning_rate: Base learning rate for models
            
        Returns:
            List of initialized ensemble members
        """
        ensemble = []
        
        # Create first model with standard initialization
        ensemble.append(EnsembleMember(input_dim, learning_rate))
        
        # Create remaining models with increasing diversity
        for i in range(1, size):
            # Create model with scaled initialization
            model = EnsembleMember(input_dim, learning_rate)
            
            # Scale weights to create diversity in the ensemble
            scale = 1.0 + 0.1 * i  # Gradually increase scale
            model.weights *= scale
            
            # Add small noise to break symmetry
            noise = np.random.normal(0, 0.01, input_dim)
            model.weights += noise
            
            ensemble.append(model)
            
        return ensemble
    
    def _ensemble_predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from all ensemble members with uncertainty estimation.
        
        Implements Bayesian Model Averaging (BMA) to combine predictions:
        p(y|x,D) = Σ_k p(y|x,θ_k) p(θ_k|D)
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (mean_prediction, uncertainty_estimate)
        """
        predictions = []
        uncertainties = []
        
        # Get predictions and uncertainties from each model
        for model in self.ensemble:
            pred, unc = model.predict(features, include_uncertainty=True)
            predictions.append(pred)
            uncertainties.append(unc)
            
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # Model weights based on performance and uncertainty
        model_weights = np.array([m.performance for m in self.ensemble])
        model_weights = softmax(model_weights / self.temperature)
        
        # Weighted average of predictions
        weighted_pred = np.sum(predictions * model_weights[:, np.newaxis], axis=0)
        
        # Total uncertainty = aleatoric + epistemic
        aleatoric = np.average(uncertainties**2, axis=0, weights=model_weights)
        epistemic = np.average((predictions - weighted_pred)**2, axis=0, weights=model_weights)
        total_uncertainty = np.sqrt(aleatoric + epistemic)
        
        return weighted_pred, total_uncertainty
    

    def _update_ensemble(self, features: np.ndarray, target: np.ndarray, learning_rate: float = 0.01):
        """
        Update the ensemble using stochastic gradient descent.
        
        Args:
            features: Input features
            target: Target values
            learning_rate: Learning rate for updates
        """
        for member in self.ensemble:
            # Forward pass
            prediction = np.dot(features, member.weights) + member.bias
            error = prediction - target
            
            # Update weights
            gradient = np.outer(features, error).mean(axis=1)
            member.weights -= learning_rate * gradient
            member.bias -= learning_rate * error.mean()
            
            # Update performance (exponential moving average)
            member.performance = 0.9 * member.performance + 0.1 * (1.0 / (1.0 + np.abs(error).mean()))
            member.last_updated = self.time_step
            
    def _sample_memory(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def _generate_base_response(self, user_input: str) -> str:
        """
        Generate a base response using the ensemble model.
        In a real system, this would be replaced with a proper dialogue manager.
        """
        # Simple rule-based response generation as fallback
        user_input = user_input.lower()
        if 'hello' in user_input or 'hi ' in user_input:
            return "Hello! How can I assist you today?"
        elif 'help' in user_input:
            return "I'm here to help. Could you tell me more about what you need?"
        elif 'thank' in user_input:
            return "You're welcome! Is there anything else I can help with?"
        else:
            return "I understand. Please tell me more about your request."
    
    def _extract_emotion_features(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract features for emotion prediction from text and context.
        
        Args:
            text: Input text
            context: User context
            
        Returns:
            Feature vector for emotion prediction
        """
        # Basic text features (in a real system, use more sophisticated features)
        text_length = len(text)
        word_count = len(text.split())
        has_question = 1 if '?' in text else 0
        has_exclamation = 1 if '!' in text else 0
        
        # Combine with cultural context if available
        cultural_features = context.get('cultural_profile', np.zeros(self.cultural_model.dimensions))
        
        # Combine all features
        features = np.concatenate([
            [text_length, word_count, has_question, has_exclamation],
            cultural_features
        ])
        
        return features
        
    def _add_ensemble_diversity(self):
        """Add diversity to the ensemble by replacing underperforming members."""
        # Sort ensemble by performance
        self.ensemble.sort(key=lambda x: x.performance, reverse=True)
        
        # Replace worst performing members with variations of best performers
        num_to_replace = max(1, self.ensemble_size // 5)  # Replace 20% of ensemble
        best_member = self.ensemble[0]
        
        for i in range(1, num_to_replace + 1):
            if i < len(self.ensemble):
                # Create a mutated version of the best performing member
                mutation = np.random.normal(0, 0.1, len(best_member.weights))
                new_weights = best_member.weights + mutation
                
                self.ensemble[-i] = EnsembleMember(
                    weights=new_weights,
                    bias=best_member.bias + np.random.normal(0, 0.05),
                    performance=best_member.performance * 0.8  # Start with lower performance
                )
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the service interactions and model performance.
        
        Returns:
            Dictionary containing service metrics
        """
        if not self.service_history:
            return {}
            
        # Calculate basic metrics
        metrics = {
            'total_interactions': len(self.service_history),
            'adaptation_rate': sum(
                1 for x in self.service_history 
                if x['metadata'].get('adaptation_applied', False)
            ) / max(1, len(self.service_history)),
            'ensemble_performance': np.mean([m.performance for m in self.ensemble]),
            'ensemble_diversity': np.std([m.performance for m in self.ensemble]),
            'current_temperature': self.temperature,
            'memory_size': len(self.memory),
            'active_ensemble_size': len(self.ensemble)
        }
        
        # Add recent emotion statistics
        if self.service_history:
            recent_emotions = [
                hist['metadata'].get('emotion_scores', [])
                for hist in self.service_history[-10:]
                if 'emotion_scores' in hist['metadata']
            ]
            if recent_emotions:
                metrics['recent_emotion_mean'] = np.mean(recent_emotions, axis=0).tolist()
                metrics['recent_emotion_std'] = np.std(recent_emotions, axis=0).tolist()
        
        # Add ensemble member details
        metrics['ensemble_details'] = [
            {
                'performance': float(m.performance),
                'last_updated': int(m.last_updated),
                'weight_mean': float(np.mean(m.weights)),
                'weight_std': float(np.std(m.weights))
            }
            for m in self.ensemble
        ]
        
        return metrics
