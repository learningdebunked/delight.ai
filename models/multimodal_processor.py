"""
Multi-modal processing module for handling text, audio, and visual inputs.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import librosa
from PIL import Image

@dataclass
class ModalityFeatures:
    """Container for features from different modalities."""
    text: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    visual: Optional[torch.Tensor] = None
    
    def to(self, device: str) -> 'ModalityFeatures':
        """Move all tensors to the specified device."""
        return ModalityFeatures(
            text=self.text.to(device) if self.text is not None else None,
            audio=self.audio.to(device) if self.audio is not None else None,
            visual=self.visual.to(device) if self.visual is not None else None
        )

class MultiModalProcessor:
    """Processes and fuses features from multiple modalities."""
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        audio_sample_rate: int = 16000,
        visual_model_name: str = "google/vit-base-patch16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the multi-modal processor.
        
        Args:
            text_model_name: Pretrained model for text processing
            audio_sample_rate: Sample rate for audio processing
            visual_model_name: Pretrained model for visual processing
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.audio_sample_rate = audio_sample_rate
        
        # Initialize text processor
        self.text_processor = AutoProcessor.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.text_model.eval()
        
        # Initialize visual processor
        self.visual_processor = AutoProcessor.from_pretrained(visual_model_name)
        self.visual_model = AutoModel.from_pretrained(visual_model_name).to(device)
        self.visual_model.eval()
        
        # Initialize fusion layer with None - will be built on first forward pass
        self.fusion = None
        self.fusion_output_dim = 512  # Desired output dimension
        self.device = device
        
    def _get_total_dim(self, features: ModalityFeatures) -> int:
        """Calculate total dimension of all provided modality features."""
        dim = 0
        if features.text is not None:
            dim += features.text.size(-1)
        if features.audio is not None:
            dim += features.audio.size(-1)
        if features.visual is not None:
            dim += features.visual.size(-1)
        return dim
        
    def _init_fusion(self, input_dim: int):
        """Initialize the fusion layer with the correct input dimension."""
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.fusion_output_dim)
        ).to(self.device)
    
    def process_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Process text input."""
        if isinstance(text, str):
            text = [text]
            
        inputs = self.text_processor(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0, :]
    
    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio file and extract features."""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
        
        # Extract mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Calculate statistics over time
        features = np.concatenate([
            np.mean(log_S, axis=1),
            np.std(log_S, axis=1),
            np.max(log_S, axis=1)
        ])
        
        return torch.tensor(features, device=self.device).float()
    
    def process_visual(self, image_path: str) -> torch.Tensor:
        """Process image file."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.visual_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.visual_model(**inputs)
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0, :].squeeze(0)
    
    def process_modalities(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> ModalityFeatures:
        """
        Process inputs from multiple modalities.
        
        Args:
            text: Input text or list of texts
            audio_path: Path to audio file
            image_path: Path to image file
            
        Returns:
            ModalityFeatures containing processed features
        """
        features = ModalityFeatures()
        
        if text is not None:
            features.text = self.process_text(text)
            
        if audio_path is not None:
            features.audio = self.process_audio(audio_path)
            
        if image_path is not None:
            features.visual = self.process_visual(image_path)
            
        return features
    
    def fuse_modalities(self, features: ModalityFeatures) -> torch.Tensor:
        """
        Fuse features from different modalities.
        
        Args:
            features: ModalityFeatures containing features to fuse
            
        Returns:
            Fused feature vector
        """
        # Collect all available features
        to_fuse = []
        batch_size = None
        
        if features.text is not None:
            # Ensure text is 2D [batch_size, features]
            text_features = features.text
            if len(text_features.shape) > 2:
                text_features = text_features.mean(dim=1)  # Average over sequence length
            to_fuse.append(text_features)
            batch_size = text_features.size(0)
            
        if features.audio is not None:
            # Ensure audio is 2D [batch_size, features]
            audio_features = features.audio
            if len(audio_features.shape) > 2:
                audio_features = audio_features.mean(dim=1)  # Average over time
            to_fuse.append(audio_features)
            if batch_size is None:
                batch_size = audio_features.size(0)
                
        if features.visual is not None:
            # Ensure visual is 2D [batch_size, features]
            visual_features = features.visual
            if len(visual_features.shape) > 2:
                if len(visual_features.shape) == 4:  # [batch, channels, height, width]
                    visual_features = visual_features.mean(dim=[2, 3])  # Global average pooling
                else:
                    visual_features = visual_features.mean(dim=1)  # Average over spatial dimensions
            to_fuse.append(visual_features)
            if batch_size is None:
                batch_size = visual_features.size(0)
            
        if not to_fuse:
            raise ValueError("No features provided for fusion")
            
        # Initialize fusion layer on first use if needed
        if self.fusion is None:
            total_dim = sum(f.size(-1) for f in to_fuse)
            self._init_fusion(total_dim)
            
        fused = torch.cat(to_fuse, dim=-1)
        return self.fusion(fused)
    
    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Tuple[ModalityFeatures, torch.Tensor]:
        """
        Process and fuse features from multiple modalities.
        
        Args:
            text: Input text or list of texts
            audio_path: Path to audio file
            image_path: Path to image file
            
        Returns:
            Tuple of (ModalityFeatures, fused_embedding)
        """
        # Process each modality with gradient disabled
        with torch.no_grad():
            features = self.process_modalities(text, audio_path, image_path)
            
            # Move features to device if needed
            features = features.to(self.device)
            
            # Fuse features
            fused = self.fuse_modalities(features)
            
        return features, fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        
        self.query = nn.Linear(query_dim, self.head_dim * num_heads)
        self.key = nn.Linear(key_dim, self.head_dim * num_heads)
        self.value = nn.Linear(value_dim, self.head_dim * num_heads)
        self.out = nn.Linear(self.head_dim * num_heads, value_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, dim)
            key: Key tensor of shape (batch_size, seq_len_k, dim)
            value: Value tensor of shape (batch_size, seq_len_v, dim)
            mask: Optional mask tensor
            
        Returns:
            Context-aware representation
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out(context)
        
        return output
