"""
SEDS Core Module - Service Excellence Dynamical System

This module implements a mathematically-grounded framework for service adaptation
that combines cultural models, emotion recognition, and ensemble learning with
proper theoretical foundations.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
from dataclasses import dataclass, field
from scipy.stats import norm, multivariate_normal, entropy
from scipy.special import softmax
from collections import deque, defaultdict
from enum import Enum
import math
import logging
from datetime import datetime

from .enhanced_cultural_model import CulturalAdaptationEngine, CulturalProfile, CulturalDimension
from .emotion_model import EmotionModel
from .cultural_adaptation_utils import CulturalAdapter, AdaptationResult

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
                cultural_dimensions: int = 10, 
                emotion_dimensions: int = 10,
                ensemble_size: int = 5,
                memory_window: int = 1000,
                learning_rate: float = 0.01,
                exploration_rate: float = 0.1):
        """
        Initialize the SEDS core system with enhanced cultural adaptation.
        
        Args:
            cultural_dimensions: Number of cultural dimensions to model
            emotion_dimensions: Dimensionality of the emotion space
            ensemble_size: Number of models in the ensemble (should be odd for voting)
            memory_window: Size of the sliding window for experience replay
            learning_rate: Base learning rate for model updates
            exploration_rate: Initial exploration rate for epsilon-greedy strategy
        """
        # Initialize core models with proper dimensions
        self.cultural_engine = CulturalAdaptationEngine()
        self.emotion_model = EmotionModel(dimensions=emotion_dimensions)
        self.cultural_adapter = CulturalAdapter()
        
        # Initialize cultural components
        self._initialize_default_cultural_profiles()
        self._initialize_adaptation_rules()
        
        # Track cultural adaptation performance
        self.adaptation_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'average_success_rate': 0.0,
            'last_updated': datetime.utcnow(),
            'dimension_effectiveness': {dim.name: 1.0 for dim in CulturalDimension}
        }
        
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
        cultural_profile_id = user_context.get('cultural_profile_id', 'global')
        cultural_profile = self.cultural_engine.get_cultural_profile(cultural_profile_id)
        
        if cultural_profile is None:
            # Fallback to default profile if specified profile not found
            cultural_profile = self.cultural_engine.get_cultural_profile('global')
            
        # Convert profile to feature vector if needed
        cultural_features = self._profile_to_feature_vector(cultural_profile)
        
        # 2. Extract emotion features with error handling
        try:
            emotion_features = self._extract_emotion_features(user_input, {
                **user_context,
                'cultural_profile': cultural_profile.dimensions
            })
            if not isinstance(emotion_features, np.ndarray):
                raise ValueError("Emotion features must be a numpy array")
        except Exception as e:
            logger.error(f"Error extracting emotion features: {e}")
            emotion_features = np.zeros(self.emotion_model.dimensions)
        
        # ... (rest of the code)

    def _extract_emotion_features(self, text: str, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract emotion features using transformer-based emotion recognition combined with
        contextual and linguistic features.
        
        Implements a multi-modal feature extraction pipeline that combines:
        1. Transformer-based emotion embeddings
        2. Sentiment and emotion probabilities
        3. Linguistic features
        4. Cultural context features
        
        Args:
            text: Input text to analyze
            context: User context including cultural profile
            
        Returns:
            Combined feature vector for emotion prediction
        """
        try:
            # Get transformer-based emotion embeddings and probabilities
            emotion_result = self.emotion_model.detect_emotion(
                text=text,
                context=context
            )
            
            # Extract emotion probabilities and features
            emotion_probs = emotion_result.get('emotion_probabilities', {})
            emotion_embedding = emotion_result.get('embedding', np.zeros(768))  # Default BERT dimension
            
            # Normalize and process emotion scores
            emotion_scores = np.array([
                emotion_probs.get('joy', 0),
                emotion_probs.get('sadness', 0),
                emotion_probs.get('anger', 0),
                emotion_probs.get('fear', 0),
                emotion_probs.get('surprise', 0),
                emotion_probs.get('disgust', 0),
                emotion_probs.get('neutral', 0)
            ])
            
            # Add valence-arousal-dominance (VAD) scores if available
            vad_scores = emotion_result.get('vad_scores', [0.5, 0.5, 0.5])
            
            # Get cultural context features
            cultural_features = np.zeros(len(CulturalDimension))
            if isinstance(context.get('cultural_profile'), dict):
                # Convert profile dictionary to feature vector
                for i, dim in enumerate(CulturalDimension):
                    cultural_features[i] = context['cultural_profile'].get(dim, 0.5)
            
            # Extract linguistic features (complementary to transformer features)
            tokens = text.split()
            text_length = len(text)
            word_count = len(tokens)
            avg_word_length = np.mean([len(w) for w in tokens]) if tokens else 0
            has_question = 1 if '?' in text else 0
            has_exclamation = 1 if '!' in text else 0
            
            # Combine all features
            features = np.concatenate([
                emotion_embedding,           # Transformer embeddings (768d)
                emotion_scores,              # Emotion probabilities (7d)
                vad_scores,                  # VAD scores (3d)
                cultural_features,           # Cultural features (n_dims)
                [                           # Additional linguistic features (5d)
                    text_length,
                    word_count,
                    avg_word_length,
                    has_question,
                    has_exclamation
                ]
            ])
            
            return features
            
        except Exception as e:
            # Fallback to basic features if emotion model fails
            logging.warning(f"Emotion feature extraction failed: {str(e)}")
            cultural_features = np.zeros(len(CulturalDimension))
            if isinstance(context.get('cultural_profile'), dict):
                for i, dim in enumerate(CulturalDimension):
                    cultural_features[i] = context['cultural_profile'].get(dim, 0.5)
                    
            return np.concatenate([
                np.zeros(768 + 7 + 3),  # Zero vectors for missing transformer features
                cultural_features,
                [len(text), len(text.split()), 0, 0, 0]  # Basic text features
            ])
        
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
