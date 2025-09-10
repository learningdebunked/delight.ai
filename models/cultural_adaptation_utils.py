"""
Utility functions for cultural adaptation in the SEDS framework.
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import random
from enum import Enum
import numpy as np
from dataclasses import dataclass

class FormalityLevel(Enum):
    VERY_FORMAL = 4
    FORMAL = 3
    NEUTRAL = 2
    INFORMAL = 1
    VERY_INFORMAL = 0

@dataclass
class AdaptationResult:
    """Result of a cultural adaptation operation."""
    adapted_text: str
    confidence: float
    applied_adaptations: List[Dict]
    metadata: Dict

class CulturalAdapter:
    """Handles cultural adaptation of text based on cultural dimensions."""
    
    def __init__(self):
        # Initialize formality dictionaries
        self.formal_phrases = {
            "hi": "hello",
            "hey": "hello",
            "what's up": "how are you",
            "thanks": "thank you",
            "thx": "thank you",
            "pls": "please",
            "plz": "please",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "must",
            "yeah": "yes",
            "yep": "yes",
            "nope": "no",
            "I'm": "I am",
            "you're": "you are",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "wouldn't": "would not"
        }
        
        # Initialize directness adaptations
        self.direct_phrases = {
            "could you possibly": "please",
            "would you mind": "please",
            "I was wondering if": "",
            "if it's not too much trouble": "",
            "I would appreciate it if": "",
            "at your earliest convenience": "now",
            "when you have a moment": ""
        }
        
        # Initialize politeness markers
        self.politeness_markers = [
            "please",
            "kindly",
            "if you don't mind",
            "I would be grateful if",
            "I would appreciate it if"
        ]
    
    def adapt_formality(self, text: str, target_level: float) -> Tuple[str, float]:
        """
        Adapt the formality level of the text.
        
        Args:
            text: Input text
            target_level: Target formality level (0.0 to 1.0)
            
        Returns:
            Tuple of (adapted_text, confidence)
        """
        if not text:
            return text, 0.0
            
        current_level = self._assess_formality(text)
        target_formality = FormalityLevel(self._map_to_formality_level(target_level))
        
        # No adaptation needed
        if current_level == target_formality:
            return text, 1.0
            
        adapted_text = text
        
        # Make more formal
        if current_level.value < target_formality.value:
            adapted_text = self._make_more_formal(text, target_formality)
        # Make less formal
        else:
            adapted_text = self._make_less_formal(text, target_formality)
            
        confidence = 1.0 - abs(self._assess_formality(adapted_text).value - target_formality.value) / 4.0
        return adapted_text, max(0.0, min(1.0, confidence))
    
    def adapt_directness(self, text: str, target_level: float) -> Tuple[str, float]:
        """
        Adapt the directness level of the text.
        
        Args:
            text: Input text
            target_level: Target directness level (0.0 to 1.0)
            
        Returns:
            Tuple of (adapted_text, confidence)
        """
        if not text:
            return text, 0.0
            
        current_level = self._assess_directness(text)
        
        # No adaptation needed
        if abs(current_level - target_level) < 0.1:
            return text, 1.0
            
        adapted_text = text
        
        # Make more direct
        if current_level < target_level:
            adapted_text = self._make_more_direct(text)
        # Make less direct
        else:
            adapted_text = self._make_less_direct(text)
            
        new_level = self._assess_directness(adapted_text)
        confidence = 1.0 - abs(new_level - target_level)
        return adapted_text, max(0.0, min(1.0, confidence))
    
    def _assess_formality(self, text: str) -> FormalityLevel:
        """Assess the formality level of the text."""
        # Simple heuristic based on contractions, slang, and word choice
        score = 0
        words = text.lower().split()
        
        # Check for contractions and informal words
        informal_indicators = sum(1 for word in words if "'" in word or word in self.formal_phrases)
        
        # Check for formal indicators (please, kindly, etc.)
        formal_indicators = sum(1 for word in words if word in ["please", "kindly", "appreciate", "grateful"])
        
        # Simple scoring
        score += 2 * formal_indicators
        score -= 2 * informal_indicators
        
        # Map score to formality level
        if score >= 3:
            return FormalityLevel.VERY_FORMAL
        elif score >= 1:
            return FormalityLevel.FORMAL
        elif score <= -3:
            return FormalityLevel.VERY_INFORMAL
        elif score <= -1:
            return FormalityLevel.INFORMAL
        else:
            return FormalityLevel.NEUTRAL
    
    def _assess_directness(self, text: str) -> float:
        """Assess the directness level of the text (0.0 to 1.0)."""
        # Simple heuristic based on politeness markers and sentence structure
        score = 0.5  # Neutral
        
        # Check for indirect phrases
        for phrase in self.direct_phrases:
            if phrase in text.lower():
                score -= 0.1
                
        # Check for direct commands
        if text.strip().endswith('.'):
            words = text.lower().split()
            if words and words[0] in ["please", "kindly"]:
                score += 0.1
            elif words and words[0] in ["can", "could", "would"] and "you" in words[1:3]:
                score -= 0.1
                
        return max(0.0, min(1.0, score))
    
    def _make_more_formal(self, text: str, target: FormalityLevel) -> str:
        """Make the text more formal."""
        result = text
        
        # Replace informal phrases with formal ones
        for informal, formal in self.formal_phrases.items():
            result = re.sub(r'\b' + re.escape(informal) + r'\b', formal, result, flags=re.IGNORECASE)
            
        # Add polite openings if very formal
        if target == FormalityLevel.VERY_FORMAL and not any(
            result.lower().startswith(marker) 
            for marker in ["dear", "to whom it may concern"]
        ):
            result = f"Dear Sir/Madam, {result}"
            
        return result
    
    def _make_less_formal(self, text: str, target: FormalityLevel) -> str:
        """Make the text less formal."""
        result = text
        
        # Remove formal openings
        result = re.sub(r'^\s*(dear\s+(sir/madam|sir|madam|mr\.?|ms\.?|mrs\.?)\s*,?\s*)', 
                       '', result, flags=re.IGNORECASE)
        
        # Replace formal phrases with informal ones (reverse mapping)
        for formal, informal in {v: k for k, v in self.formal_phrases.items()}.items():
            result = re.sub(r'\b' + re.escape(formal) + r'\b', informal, result, flags=re.IGNORECASE)
            
        return result.strip()
    
    def _make_more_direct(self, text: str) -> str:
        """Make the text more direct."""
        result = text.lower()
        
        # Replace indirect phrases with direct ones
        for indirect, direct in self.direct_phrases.items():
            if direct:  # Only replace if there's a direct equivalent
                result = result.replace(indirect, direct)
            else:
                # Remove the indirect phrase
                result = result.replace(indirect, '')
                
        # Remove unnecessary politeness markers
        for marker in self.politeness_markers:
            result = result.replace(marker, '')
            
        # Clean up any double spaces or leading/trailing spaces
        result = ' '.join(result.split())
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
            
        return result
    
    def _make_less_direct(self, text: str) -> str:
        """Make the text less direct."""
        if not text.strip():
            return text
            
        result = text.lower()
        
        # Add politeness markers if not already present
        has_politeness = any(marker in result for marker in self.politeness_markers)
        if not has_politeness and not result.startswith(('please', 'kindly')):
            # Add a random politeness marker at the beginning
            marker = random.choice(self.politeness_markers)
            result = f"{marker} {result}"
            
        # Make questions more indirect
        if '?' in result and not any(phrase in result for phrase in ["could you", "would you"]):
            result = result.replace('?', ' please?')
            
        # Clean up any double spaces or leading/trailing spaces
        result = ' '.join(result.split())
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
            
        return result
    
    def _map_to_formality_level(self, value: float) -> int:
        """Map a float value (0.0 to 1.0) to a FormalityLevel."""
        if value >= 0.8:
            return FormalityLevel.VERY_FORMAL.value
        elif value >= 0.6:
            return FormalityLevel.FORMAL.value
        elif value >= 0.4:
            return FormalityLevel.NEUTRAL.value
        elif value >= 0.2:
            return FormalityLevel.INFORMAL.value
        else:
            return FormalityLevel.VERY_INFORMAL.value
