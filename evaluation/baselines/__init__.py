"""Baseline models for comparison."""

from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)

class BaselineModel(nn.Module):
    """Base class for baseline models."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        """Initialize the baseline model.
        
        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of output labels
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model and tokenizer
        self.config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Make predictions on a list of texts."""
        self.eval()
        all_preds = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_preds)
    
    @property
    def device(self):
        """Get the device the model is on."""
        return next(self.parameters()).device
    
    def save_pretrained(self, output_dir: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class CulturalBaseline(BaselineModel):
    """Baseline model for cultural adaptation tasks."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_cultural_dims: int = 6):
        """Initialize the cultural baseline model.
        
        Args:
            model_name: Name of the pre-trained model
            num_cultural_dims: Number of cultural dimensions to predict
        """
        super().__init__(model_name, num_labels=num_cultural_dims)
        
        # Replace the classifier head for regression
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, num_cultural_dims),
            nn.Sigmoid()  # Output between 0 and 1 for each dimension
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with MSE loss for regression."""
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[1]
        logits = self.model.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        })


class EmotionBaseline(BaselineModel):
    """Baseline model for emotion detection tasks."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_emotions: int = 6):
        """Initialize the emotion baseline model.
        
        Args:
            model_name: Name of the pre-trained model
            num_emotions: Number of emotion categories
        """
        super().__init__(model_name, num_labels=num_emotions)
        
        # Replace the classifier head for multi-label classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, num_emotions),
            nn.Sigmoid()  # Multi-label probabilities
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with BCE loss for multi-label classification."""
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs[1]
        logits = self.model.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        })
