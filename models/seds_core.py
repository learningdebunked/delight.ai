import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from scipy.stats import norm, multivariate_normal
from collections import deque
from .cultural_model import CulturalModel
from .emotion_model import EmotionModel

@dataclass
class EnsembleMember:
    """Represents a single model in the ensemble."""
    weights: np.ndarray
    bias: float
    performance: float = 1.0
    last_updated: float = 0.0


class SEDSCore:
    """
    Core implementation of the Service Excellence Dynamical System (SEDS).
    Integrates cultural and emotion models with stochastic ensemble dynamics.
    """
    
    def __init__(self, 
                cultural_dimensions: int = 25, 
                emotion_dimensions: int = 50,
                ensemble_size: int = 5,
                memory_window: int = 100):
        """
        Initialize the SEDS core with ensemble capabilities.
        
        Args:
            cultural_dimensions: Number of cultural dimensions to model
            emotion_dimensions: Dimensionality of the emotion space
            ensemble_size: Number of models in the ensemble
            memory_window: Size of the sliding window for experience replay
        """
        self.cultural_model = CulturalModel(dimensions=cultural_dimensions)
        self.emotion_model = EmotionModel(dimensions=emotion_dimensions)
        self.service_history = []
        self.ensemble_size = ensemble_size
        self.ensemble = self._initialize_ensemble(ensemble_size, cultural_dimensions)
        self.memory = deque(maxlen=memory_window)
        self.time_step = 0
        self.temperature = 1.0  # Controls exploration vs exploitation
        
    def process_interaction(self, user_input: str, user_context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user interaction using stochastic ensemble dynamics.
        
        Args:
            user_input: User's text input
            user_context: Dictionary containing user context including cultural profile
            
        Returns:
            Tuple of (response, metadata)
        """
        self.time_step += 1
        
        # Extract cultural context
        cultural_profile = user_context.get('cultural_profile', np.zeros(self.cultural_model.dimensions))
        
        # Analyze emotion using ensemble prediction
        emotion_features = self._extract_emotion_features(user_input, user_context)
        emotion_scores = self._ensemble_predict(emotion_features)
        emotion_state = self.emotion_model.update_emotion_state(emotion_scores)
        
        # Generate base response using ensemble
        base_response = self._generate_base_response(user_input)
        
        # Apply cultural adaptation with stochastic variation
        system_culture = np.ones(self.cultural_model.dimensions) * 0.5
        
        # Add stochastic perturbation to cultural adaptation
        noise = np.random.normal(0, 0.1, self.cultural_model.dimensions)
        perturbed_culture = np.clip(cultural_profile + noise, 0, 1)
        
        adapted_response = self.cultural_model.adapt_response(
            base_response,
            source_culture=system_culture,
            target_culture=perturbed_culture
        )
        
        # Prepare metadata with additional ensemble information
        metadata = {
            'emotion_scores': emotion_scores.tolist(),
            'emotion_state': emotion_state.tolist(),
            'cultural_profile': cultural_profile.tolist(),
            'base_response': base_response,
            'adaptation_applied': adapted_response != base_response,
            'ensemble_performance': [m.performance for m in self.ensemble],
            'temperature': self.temperature,
            'time_step': self.time_step
        }
        
        # Store interaction in memory for experience replay
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
        
    def _initialize_ensemble(self, size: int, dimensions: int) -> List[EnsembleMember]:
        """Initialize the ensemble with random models."""
        return [
            EnsembleMember(
                weights=np.random.normal(0, 0.1, dimensions),
                bias=0.0,
                performance=1.0
            )
            for _ in range(size)
        ]

    def _ensemble_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble with stochastic selection.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Weighted average of ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for member in self.ensemble:
            pred = np.dot(features, member.weights) + member.bias
            predictions.append(pred)
            
        # Apply softmax to member performances for weighting
        performances = np.array([m.performance for m in self.ensemble])
        weights = np.exp(performances / self.temperature)
        weights /= weights.sum()
        
        # Weighted average of predictions
        return np.average(predictions, axis=0, weights=weights)

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
