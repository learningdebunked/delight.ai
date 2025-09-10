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
    
    # Performance optimization settings
    use_amp: bool = True  # Automatic Mixed Precision
    use_gradient_checkpointing: bool = True
    use_8bit_quantization: bool = torch.cuda.is_available()
    use_torch_compile: bool = torch.__version__ >= '2.0.0'  # Requires PyTorch 2.0+
    use_memory_efficient_attention: bool = True
    
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
        """Initialize the emotion detection models with performance optimizations."""
        try:
            logger.info(f"Initializing emotion model: {self.config.model_type}")
            
            # Initialize tokenizer and base model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Enable gradient checkpointing if specified
            if self.config.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint_sequential
                self.checkpoint_sequential = checkpoint_sequential
                logger.info("Enabled gradient checkpointing")
            
            # Set up mixed precision training
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
            self.amp_dtype = getattr(torch, self.config.amp_dtype) if hasattr(torch, self.config.amp_dtype) else torch.float16
            
            # Initialize model with appropriate configuration
            if self.config.model_type == EmotionModelType.ADVANCED:
                config = BertConfig.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.emotion_labels),
                    output_hidden_states=True,
                    output_attentions=True,
                    hidden_dropout_prob=0.2,
                    attention_probs_dropout_prob=0.2,
                    # Memory efficient attention
                    attention_probs_dropout_prob=0.1 if self.config.use_memory_efficient_attention else 0.0,
                    use_memory_efficient_attention=self.config.use_memory_efficient_attention
                )
                
                # Load model with memory optimizations
                with torch.device_scope(self.device):
                    self.model = BertModel.from_pretrained(
                        self.config.model_name,
                        config=config,
                        torch_dtype=self.amp_dtype if self.config.use_amp else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto" if torch.cuda.device_count() > 1 else None
                    )
                
                # Add custom classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(config.hidden_size, 256),
                    nn.GELU(),
                    nn.LayerNorm(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(self.config.emotion_labels))
                ).to(self.device)
                
                # Initialize weights
                self.classifier.apply(self._init_weights)
                
                # Apply gradient checkpointing if enabled
                if self.config.use_gradient_checkpointing:
                    self.model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing for the model")
                
            else:
                # Standard pre-trained model with optimizations
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.emotion_labels),
                    torch_dtype=self.amp_dtype if self.config.use_amp else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.device_count() > 1 else None
                )
            
            # Apply optimizations
            self._apply_model_optimizations()
            
            # Initialize uncertainty estimator with optimizations
            self.uncertainty_estimator = UncertaintyEstimator(
                self.model,
                num_passes=self.config.mc_dropout_passes,
                use_amp=self.config.use_amp
            )
            
            # Initialize sentence encoder with optimizations
            self.sentence_encoder = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            
            # Compile model if supported and enabled
            if self.config.use_torch_compile and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compilation with torch.compile completed")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            logger.info("Successfully initialized emotion models")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion models: {e}")
            raise
    
    def _init_weights(self, module):
        """Initialize weights for custom layers with optimized initialization."""
        if isinstance(module, nn.Linear):
            # Kaiming initialization with fan_in mode for ReLU/GELU
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='gelu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm with 1.0 for weight and 0.0 for bias
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with smaller scale for better training stability
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def _apply_model_optimizations(self):
        """Apply various model optimizations."""
        # Move model to device with memory optimizations
        self.model = self.model.to(self.device)
        
        # Apply 8-bit quantization if enabled and supported
        if self.config.use_8bit_quantization and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Linear8bitLt
                
                # Replace linear layers with 8-bit quantized versions
                for name, module in self.model.named_children():
                    if isinstance(module, nn.Linear):
                        # Skip classification head from quantization
                        if 'classifier' not in name and 'pooler' not in name:
                            quantized_layer = Linear8bitLt(
                                module.in_features,
                                module.out_features,
                                bias=module.bias is not None,
                                has_fp16_weights=False
                            )
                            quantized_layer.weight = bnb.nn.Int8Params(
                                module.weight.data,
                                requires_grad=True
                            )
                            if module.bias is not None:
                                quantized_layer.bias = nn.Parameter(module.bias.data.clone())
                            setattr(self.model, name, quantized_layer)
                logger.info("Applied 8-bit quantization to model weights")
                
            except ImportError:
                logger.warning("bitsandbytes not available, skipping 8-bit quantization")
        
        # Apply gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Enable memory-efficient attention if available
        if self.config.use_memory_efficient_attention and hasattr(torch.backends, 'xformers'):
            try:
                from xformers.ops import memory_efficient_attention
                torch.backends.xformers.enable_mem_efficient_attention()
                logger.info("Enabled memory-efficient attention")
            except ImportError:
                logger.warning("xformers not available, using standard attention")
    
    def _quantize_model(self):
        """Apply dynamic quantization to the model."""
        if not self.config.dynamic_quantization or not torch.cuda.is_available():
            return
            
        logger.info("Applying dynamic quantization to the model")
        try:
            # Quantize the model with dynamic quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # Quantize only linear layers
                dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply dynamic quantization: {e}")
    
    def train_step(self, batch, optimizer):
        """Perform a single training step with optimizations."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
        
        # Scale loss and backpropagate
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate scheduler if available
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
        
        self.global_step += 1
        return loss.item()
            
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
    def _init_multimodal_models(self):
        """Initialize models for multimodal emotion detection."""
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        from transformers import ViTFeatureExtractor, ViTModel
        
        # Initialize audio model (Wav2Vec 2.0)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.audio_model.eval()
        
        # Initialize visual model (ViT)
        self.visual_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.visual_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
        self.visual_model.eval()
        
        # Cross-modal attention layers
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True).to(self.device)
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights for text, audio, visual
        
    def _process_audio(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process audio input and extract emotion-related features."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
            
        # Process audio through Wav2Vec2
        with torch.no_grad():
            inputs = self.audio_processor(
                audio.squeeze().numpy(), 
                return_tensors="pt", 
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            
            outputs = self.audio_model(**inputs)
            # Use mean pooling over time dimension
            audio_features = outputs.last_hidden_state.mean(dim=1)
            
        return audio_features
    
    def _process_visual(self, visual: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process visual input and extract emotion-related features."""
        if isinstance(visual, np.ndarray):
            if visual.max() > 1.0:  # Assuming 0-255 range
                visual = visual.astype(np.float32) / 255.0
            visual = torch.from_numpy(visual).permute(2, 0, 1)  # HWC to CHW
            
        if visual.dim() == 3:
            visual = visual.unsqueeze(0)  # Add batch dimension
            
        # Process visual through ViT
        with torch.no_grad():
            inputs = self.visual_processor(
                images=visual,
                return_tensors="pt"
            )['pixel_values'].to(self.device)
            
            outputs = self.visual_model(pixel_values=inputs)
            visual_features = outputs.last_hidden_state[:, 0]  # [CLS] token
            
        return visual_features
    
    def _fuse_modalities(self, text_features: torch.Tensor, 
                        audio_features: Optional[torch.Tensor] = None,
                        visual_features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Fuse features from different modalities using attention."""
        # Project all features to same dimension if needed
        text_features = text_features.unsqueeze(1)  # [batch, 1, dim]
        
        # Prepare modality features for attention
        modality_features = [text_features]
        if audio_features is not None:
            audio_features = audio_features.unsqueeze(1)
            modality_features.append(audio_features)
        if visual_features is not None:
            visual_features = visual_features.unsqueeze(1)
            modality_features.append(visual_features)
            
        # Stack and apply attention
        stacked_features = torch.cat(modality_features, dim=1)  # [batch, num_modalities, dim]
        
        # Cross-attention between modalities
        attended, _ = self.cross_attention(
            stacked_features,  # query
            stacked_features,  # key
            stacked_features,  # value
            need_weights=False
        )
        
        # Weighted sum of attended features
        weights = torch.softmax(self.modality_weights[:len(modality_features)], dim=0)
        fused_features = (attended * weights.view(1, -1, 1)).sum(dim=1)
        
        # Predict emotions from fused features
        logits = self.classifier(fused_features)
        probs = torch.softmax(logits, dim=-1)
        
        # Convert to emotion dictionary
        emotions = {
            label: probs[0, i].item() 
            for i, label in enumerate(self.config.emotion_labels)
        }
        
        return emotions
    
    def _detect_multimodal_emotion(self, 
                                 text: Optional[str] = None,
                                 audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                 visual: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Detect emotions from multimodal inputs using transformer-based models.
        
        Implements a sophisticated multimodal fusion approach with:
        1. Text processing using the base transformer
        2. Audio processing with Wav2Vec2
        3. Visual processing with Vision Transformer (ViT)
        4. Cross-modal attention for feature fusion
        
        Args:
            text: Input text for emotion analysis
            audio: Raw audio waveform (numpy array or torch.Tensor)
            visual: Input image (numpy array or torch.Tensor)
            
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        if not hasattr(self, 'audio_model'):
            self._init_multimodal_models()
        
        # Process text if available
        text_features = None
        if text:
            inputs = self._prepare_text_inputs(text)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                text_features = outputs.hidden_states[-1][:, 0]  # [CLS] token
        
        # Process audio if available
        audio_features = None
        if audio is not None:
            audio_features = self._process_audio(audio)
        
        # Process visual if available
        visual_features = None
        if visual is not None:
            visual_features = self._process_visual(visual)
        
        # Fuse modalities and predict emotions
        emotions = self._fuse_modalities(
            text_features=text_features if text_features is not None else torch.zeros(1, 768).to(self.device),
            audio_features=audio_features,
            visual_features=visual_features
        )
        
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
