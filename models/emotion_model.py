import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    pipeline
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelType(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"

@dataclass
class EmotionConfig:
    model_name: str = "SamLowe/roberta-base-go_emotions"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    threshold: float = 0.1
    top_k: int = 5
    model_type: EmotionModelType = EmotionModelType.TEXT

class EmotionModel:
    """
    Enhanced emotion detection and modeling component of the SEDS framework.
    Supports multiple emotion models and multimodal inputs.
    """
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        """
        Initialize the enhanced emotion model.
        
        Args:
            config: Configuration for the emotion model
        """
        self.config = config or EmotionConfig()
        self.device = self.config.device
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.sentence_encoder = None
        
        self._initialize_models()
        
        # Initialize emotion state
        self.emotion_state = np.zeros(self.config.dimensions)
        self.emotion_history = []
        
        logger.info(f"Initialized EmotionModel on device: {self.device}")
        
    def _initialize_models(self):
        """Initialize the emotion detection models based on configuration."""
        try:
            logger.info(f"Loading emotion model: {self.config.model_name}")
            
            if self.config.model_type == EmotionModelType.TEXT:
                # Initialize text-based emotion model
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name
                ).to(self.device)
                self.model.eval()
                
            elif self.config.model_type == EmotionModelType.MULTIMODAL:
                # Initialize multimodal model (e.g., for text + audio/visual)
                self.processor = AutoProcessor.from_pretrained(self.config.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name
                ).to(self.device)
                self.model.eval()
            
            # Initialize sentence encoder for emotion state
            self.sentence_encoder = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
            
            logger.info("Successfully loaded emotion models")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion models: {e}")
            raise
            
    def detect_emotion(self, 
                      text: Optional[str] = None,
                      audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      visual: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Detect emotions from text, audio, or visual inputs.
        
        Args:
            text: Input text to analyze
            audio: Audio signal (for multimodal models)
            visual: Visual input (for multimodal models)
            
        Returns:
            Dictionary of emotion scores
        """
        if text is None and audio is None and visual is None:
            raise ValueError("At least one input modality (text, audio, or visual) must be provided")
            
        try:
            if self.config.model_type == EmotionModelType.TEXT and text:
                return self._detect_text_emotion(text)
                
            elif self.config.model_type == EmotionModelType.MULTIMODAL:
                return self._detect_multimodal_emotion(text, audio, visual)
                
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}", exc_info=True)
            return {}
    
    def _detect_text_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotions from text input."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get emotion probabilities
        probs = F.softmax(outputs.logits, dim=-1)[0]
        
        # Get top-k emotions
        top_probs, top_indices = torch.topk(probs, k=self.config.top_k)
        
        # Map to emotion labels
        emotions = {}
        for idx, prob in zip(top_indices, top_probs):
            if prob >= self.config.threshold:
                label = self.model.config.id2label[idx.item()]
                emotions[label] = prob.item()
                
        return emotions
        
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
