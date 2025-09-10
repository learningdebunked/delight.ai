"""
Enhanced Emotion Model for SEDS Framework

Implements a state-of-the-art emotion detection system with:
- Transformer-based emotion classification
- Temporal modeling of emotion states
- Uncertainty quantification
- Multimodal fusion capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    BertModel,
    BertConfig,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import math
from scipy.stats import entropy
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelType(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"
    ADVANCED = "advanced"  # New model type with temporal modeling

class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"

@dataclass
class EmotionConfig:
    # Model configuration
    model_name: str = "bert-base-uncased"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 128
    model_type: EmotionModelType = EmotionModelType.ADVANCED
    
    # Emotion detection
    emotion_labels: List[str] = field(default_factory=lambda: list(EmotionType._value2member_map_.keys()))
    threshold: float = 0.2  # Minimum probability to consider an emotion present
    top_k: int = 3  # Number of top emotions to return
    
    # Temporal modeling
    history_window: int = 10  # Number of past states to consider
    decay_factor: float = 0.9  # Decay rate for past emotions
    
    # Uncertainty estimation
    mc_dropout_passes: int = 5  # Number of forward passes for MC Dropout
    uncertainty_threshold: float = 0.3  # Threshold for high uncertainty
    
    # Training parameters (if fine-tuning)
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100

class EmotionState:
    """Class to manage the temporal state of emotions."""
    
    def __init__(self, num_emotions: int, window_size: int = 10, decay: float = 0.9):
        self.num_emotions = num_emotions
        self.window_size = window_size
        self.decay = decay
        self.history = deque(maxlen=window_size)
        self.current_state = np.zeros(num_emotions)
        self.uncertainty = 1.0  # Initial high uncertainty
        
    def update(self, emotion_probs: np.ndarray, uncertainty: float) -> None:
        """Update the emotion state with new observations."""
        # Apply decay to current state
        self.current_state *= self.decay
        
        # Add new observation with uncertainty weighting
        confidence = 1.0 - min(uncertainty, 1.0)
        self.current_state = self.current_state * (1 - confidence) + emotion_probs * confidence
        
        # Normalize to maintain probability distribution
        self.current_state = self.current_state / (self.current_state.sum() + 1e-8)
        
        # Update history
        self.history.append(self.current_state.copy())
        self.uncertainty = uncertainty
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the current dominant emotion and its intensity."""
        idx = np.argmax(self.current_state)
        return self.config.emotion_labels[idx], float(self.current_state[idx])
    
    def get_emotion_trend(self, window: int = 5) -> Dict[str, float]:
        """Calculate the trend of each emotion over the recent window."""
        if len(self.history) < 2:
            return {e: 0.0 for e in self.config.emotion_labels}
            
        recent = np.array(list(self.history)[-window:])
        if len(recent) < 2:
            return {self.config.emotion_labels[i]: 0.0 
                   for i in range(self.num_emotions)}
            
        # Calculate slope using linear regression
        x = np.arange(len(recent))
        trends = []
        for i in range(self.num_emotions):
            if np.all(recent[:, i] == recent[0, i]):
                trends.append(0.0)
                continue
            z = np.polyfit(x, recent[:, i], 1)
            trends.append(z[0])
            
        return {self.config.emotion_labels[i]: float(trends[i]) 
               for i in range(self.num_emotions)}


class UncertaintyEstimator:
    """Handles uncertainty estimation using MC Dropout and ensemble methods."""
    
    def __init__(self, model: nn.Module, num_passes: int = 5):
        self.model = model
        self.num_passes = num_passes
        
    def estimate(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """Estimate uncertainty using MC Dropout."""
        self.model.eval()
        self.model.enable_dropout()
        
        # Get multiple predictions
        with torch.no_grad():
            outputs = [self.model(**inputs).logits.softmax(dim=-1) 
                      for _ in range(self.num_passes)]
            
        # Stack predictions and compute statistics
        predictions = torch.stack(outputs)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Compute uncertainty as average standard deviation
        uncertainty = std_pred.mean().item()
        
        return mean_pred.cpu().numpy(), uncertainty


class EmotionModel(nn.Module):
    """
    Advanced Emotion Detection and Modeling System
    
    Features:
    - Transformer-based emotion classification
    - Temporal modeling of emotion states
    - Uncertainty quantification
    - Multimodal fusion capabilities
    """
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        """
        Initialize the advanced emotion model.
        
        Args:
            config: Configuration for the emotion model
        """
        super().__init__()
        self.config = config or EmotionConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize models and components
        self.tokenizer = None
        self.model = None
        self.uncertainty_estimator = None
        self.sentence_encoder = None
        
        # Initialize emotion state
        self.emotion_state = EmotionState(
            num_emotions=len(self.config.emotion_labels),
            window_size=self.config.history_window,
            decay=self.config.decay_factor
        )
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Initialized EmotionModel (type: {self.config.model_type}) on device: {self.device}")
        
    def _initialize_models(self):
        """Initialize the emotion detection models based on configuration."""
        try:
            logger.info(f"Initializing emotion model: {self.config.model_type}")
            
            # Initialize tokenizer and base model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            if self.config.model_type == EmotionModelType.ADVANCED:
                # Advanced model with custom architecture
                config = BertConfig.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.emotion_labels),
                    output_hidden_states=True,
                    output_attentions=True,
                    hidden_dropout_prob=0.2,
                    attention_probs_dropout_prob=0.2
                )
                self.model = BertModel.from_pretrained(
                    self.config.model_name,
                    config=config
                )
                
                # Add custom classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(config.hidden_size, 256),
                    nn.GELU(),
                    nn.LayerNorm(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(self.config.emotion_labels))
                )
                
                # Initialize weights
                self.classifier.apply(self._init_weights)
                
            else:
                # Standard pre-trained model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.emotion_labels)
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Initialize uncertainty estimator
            self.uncertainty_estimator = UncertaintyEstimator(
                self.model,
                num_passes=self.config.mc_dropout_passes
            )
            
            # Initialize sentence encoder for contextual understanding
            self.sentence_encoder = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
            
            logger.info("Successfully initialized emotion models")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion models: {e}")
            raise
    
    def _init_weights(self, module):
        """Initialize weights for custom layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.config.model_type == EmotionModelType.ADVANCED:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use [CLS] token representation
            sequence_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(sequence_output)
            return logits
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs.logits
    
    def detect_emotion(
        self, 
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        visual: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect emotions with uncertainty estimation and temporal modeling.
        
        Args:
            text: Input text for emotion analysis
            audio: Optional audio features for multimodal analysis
            visual: Optional visual features for multimodal analysis
            context: Additional context for emotion analysis
            
        Returns:
            Dictionary containing:
            - emotions: Dict of emotion probabilities
            - dominant_emotion: Tuple of (emotion, probability)
            - uncertainty: Uncertainty score (0-1)
            - trend: Trend of each emotion over time
            - state: Current emotion state vector
        """
        if text is None and audio is None and visual is None:
            raise ValueError("At least one input modality must be provided")
        
        # Process input based on available modalities
        if text is not None:
            inputs = self._prepare_text_inputs(text)
        else:
            inputs = self._prepare_multimodal_inputs(audio, visual)
        
        # Get model predictions with uncertainty estimation
        with torch.no_grad():
            if self.config.model_type == EmotionModelType.ADVANCED:
                # Use MC Dropout for uncertainty estimation
                emotion_probs, uncertainty = self.uncertainty_estimator.estimate(inputs)
            else:
                # Standard forward pass
                outputs = self.model(**inputs)
                emotion_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                uncertainty = 0.0  # No uncertainty estimation for standard models
        
        # Convert to dictionary format
        emotion_dict = {
            self.config.emotion_labels[i]: float(emotion_probs[0][i]) 
            for i in range(len(self.config.emotion_labels))
        }
        
        # Update emotion state
        self.emotion_state.update(emotion_probs[0], uncertainty)
        
        # Get dominant emotion and trends
        dominant_emotion, confidence = self.emotion_state.get_dominant_emotion()
        trends = self.emotion_state.get_emotion_trend()
        
        return {
            'emotions': emotion_dict,
            'dominant_emotion': (dominant_emotion, confidence),
            'uncertainty': float(uncertainty),
            'trend': trends,
            'state': self.emotion_state.current_state.tolist(),
            'high_uncertainty': uncertainty > self.config.uncertainty_threshold
        }
    
    def _prepare_text_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare text inputs for the model."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True
        ).to(self.device)
    def _detect_multimodal_emotion(self, 
                                 text: Optional[str] = None,
                                 audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                 visual: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Detect emotions from multimodal inputs.
        This is a placeholder implementation that can be extended with specific models.
        """
        # This would be implemented based on the specific multimodal model being used
        # For example, using Wav2Vec2 for audio and ViT for images
        
        # Placeholder implementation
        emotions = {}
        
        if text:
            text_emotions = self._detect_text_emotion(text)
            emotions.update(text_emotions)
            
        # Add audio and visual processing here
        # For example:
        # if audio is not None:
        #     audio_emotions = self._process_audio(audio)
        #     emotions = self._fuse_emotions(emotions, audio_emotions)
        # 
        # if visual is not None:
        #     visual_emotions = self._process_visual(visual)
        #     emotions = self._fuse_emotions(emotions, visual_emotions)
            
        return emotions
    
    def update_emotion_state(self, 
                           new_emotions: Dict[str, float], 
                           decay_factor: float = 0.9,
                           max_history: int = 100) -> np.ndarray:
        """
        Update the internal emotion state based on new observations with enhanced tracking.
        
        Args:
            new_emotions: Dictionary of new emotion observations
            decay_factor: How quickly previous emotions decay (0-1)
            max_history: Maximum number of emotion states to keep in history
            
        Returns:
            Updated emotion state vector
        """
        if not new_emotions:
            return self.emotion_state
            
        try:
            # Convert emotion dict to a semantic embedding
            emotion_text = " ".join(f"{k}:{v:.2f}" for k, v in new_emotions.items())
            emotion_embedding = self.sentence_encoder.encode(
                emotion_text,
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu().numpy()
            
            # Update history
            self.emotion_history.append({
                'timestamp': datetime.now(),
                'emotions': new_emotions.copy(),
                'embedding': emotion_embedding
            })
            
            # Keep only recent history
            if len(self.emotion_history) > max_history:
                self.emotion_history = self.emotion_history[-max_history:]
            
            # Calculate weighted emotion state
            weights = np.array([decay_factor ** i for i in range(len(self.emotion_history))])
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average of historical embeddings
            historical_embeddings = np.array([h['embedding'] for h in self.emotion_history])
            self.emotion_state = np.average(historical_embeddings, axis=0, weights=weights)
            
            return self.emotion_state.copy()
            
        except Exception as e:
            logger.error(f"Error updating emotion state: {e}", exc_info=True)
            return self.emotion_state
    
    def get_emotion_summary(self, top_k: int = 5) -> Dict[str, float]:
        """
        Get a summary of the current emotional state with enhanced analysis.
        
        Args:
            top_k: Number of top emotions to return
            
        Returns:
            Dictionary of dominant emotions and their intensities
        """
        if not self.emotion_history:
            return {}
            
        try:
            # Aggregate recent emotions
            emotion_counts = {}
            emotion_scores = {}
            
            for entry in self.emotion_history[-50:]:  # Last 50 emotions
                for emotion, score in entry['emotions'].items():
                    if emotion not in emotion_counts:
                        emotion_counts[emotion] = 0
                        emotion_scores[emotion] = 0.0
                    emotion_counts[emotion] += 1
                    emotion_scores[emotion] += score
            
            # Calculate average scores
            avg_scores = {
                k: v / emotion_counts[k] 
                for k, v in emotion_scores.items()
            }
            
            # Get top-k emotions
            sorted_emotions = sorted(
                avg_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            return dict(sorted_emotions)
            
        except Exception as e:
            logger.error(f"Error generating emotion summary: {e}", exc_info=True)
            return {}
    
    def get_emotion_trends(self, window_size: int = 5) -> Dict[str, List[float]]:
        """
        Get emotion trends over time.
        
        Args:
            window_size: Size of the rolling window for smoothing
            
        Returns:
            Dictionary mapping emotions to their trend values
        """
        if not self.emotion_history:
            return {}
            
        # Initialize trend data structure
        all_emotions = set()
        for entry in self.emotion_history:
            all_emotions.update(entry['emotions'].keys())
            
        trends = {e: [] for e in all_emotions}
        
        # Extract time series data
        for entry in self.emotion_history:
            for emotion in all_emotions:
                trends[emotion].append(entry['emotions'].get(emotion, 0.0))
        
        # Apply simple moving average
        if window_size > 1:
            window = np.ones(window_size) / window_size
            for emotion in trends:
                trends[emotion] = np.convolve(
                    trends[emotion], 
                    window, 
                    mode='same'
                ).tolist()
        
        return trends
