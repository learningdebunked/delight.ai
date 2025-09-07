import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CulturalDimension(str, Enum):
    """Core cultural dimensions based on Hofstede's model and extensions."""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    MASCULINITY = "masculinity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"
    MONOCHRONIC_TIME = "monochronic_time"
    HIGH_CONTEXT = "high_context"
    UNIVERSALISM = "universalism"
    SPECIFICITY = "specificity"
    ACHIEVEMENT = "achievement"
    AFFECTIVE_EXPRESSIVENESS = "affective_expressiveness"
    INSTRUMENTALITY = "instrumentality"
    FACE_SAVING = "face_saving"
    COLLECTIVE_RESPONSIBILITY = "collective_responsibility"
    HIERARCHY = "hierarchy"
    FORMALITY = "formality"
    DIRECTNESS = "directness"
    EMOTIONAL_EXPRESSION = "emotional_expression"
    RELATIONSHIP_FOCUS = "relationship_focus"
    TIME_ORIENTATION = "time_orientation"
    COMMUNICATION_STYLE = "communication_style"
    DECISION_MAKING = "decision_making"
    TRUST_BASIS = "trust_basis"

@dataclass
class RegionProfile:
    """Profile for a specific geographic or cultural region."""
    region_id: str
    name: str
    description: str = ""
    cultural_dimensions: Dict[CulturalDimension, float] = field(default_factory=dict)
    adaptation_rules: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'region_id': self.region_id,
            'name': self.name,
            'description': self.description,
            'cultural_dimensions': {k.value: v for k, v in self.cultural_dimensions.items()},
            'adaptation_rules': self.adaptation_rules
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegionProfile':
        """Create from dictionary."""
        dimensions = {
            CulturalDimension(k): v 
            for k, v in data.get('cultural_dimensions', {}).items()
        }
        return cls(
            region_id=data['region_id'],
            name=data['name'],
            description=data.get('description', ''),
            cultural_dimensions=dimensions,
            adaptation_rules=data.get('adaptation_rules', {})
        )

@dataclass
class AdaptationRule:
    """Rule for cultural adaptation."""
    name: str
    condition: str  # Python expression that evaluates to True/False
    action: str     # Python code to execute when condition is True
    priority: int = 0
    description: str = ""
    
    def evaluate(self, context: Dict) -> bool:
        """Evaluate the rule's condition in the given context."""
        try:
            return bool(eval(self.condition, {}, context))
        except Exception as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False
    
    def apply(self, context: Dict) -> Dict:
        """Apply the rule's action in the given context."""
        try:
            local_vars = context.copy()
            exec(self.action, {}, local_vars)
            return local_vars.get('result', {})
        except Exception as e:
            logger.error(f"Error applying action: {e}")
            return {}

class CulturalModel:
    """
    Enhanced cultural model for the SEDS framework with regional adaptations.
    Handles cultural dimensions, regional profiles, and adaptation rules.
    """
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the cultural model with optional model directory.
        
        Args:
            model_dir: Directory to save/load model files (optional)
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.region_profiles: Dict[str, RegionProfile] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.default_region = "global"
        
        # Initialize default region profile
        self._initialize_default_region()
        
        # Load saved models if directory exists
        if self.model_dir and self.model_dir.exists():
            self.load_models()
    
    def _initialize_default_region(self):
        """Initialize the default global region profile."""
        default_dimensions = {
            dim: 0.5  # Neutral value for all dimensions
            for dim in CulturalDimension
        }
        
        self.region_profiles[self.default_region] = RegionProfile(
            region_id=self.default_region,
            name="Global Default",
            description="Default cultural profile with neutral values",
            cultural_dimensions=default_dimensions,
            adaptation_rules={
                "default": "result = {'adaptation_level': 'minimal'}"
            }
        )
        
    def add_region_profile(self, profile: RegionProfile) -> None:
        """
        Add or update a region profile.
        
        Args:
            profile: RegionProfile instance to add/update
        """
        self.region_profiles[profile.region_id] = profile
        logger.info(f"Added/updated region profile: {profile.name} ({profile.region_id})")
    
    def get_region_profile(self, region_id: str) -> Optional[RegionProfile]:
        """
        Get a region profile by ID.
        
        Args:
            region_id: ID of the region to retrieve
            
        Returns:
            RegionProfile if found, else None
        """
        return self.region_profiles.get(region_id)
    
    def add_adaptation_rule(self, rule: AdaptationRule) -> None:
        """
        Add or update an adaptation rule.
        
        Args:
            rule: AdaptationRule instance to add/update
        """
        self.adaptation_rules[rule.name] = rule
        logger.info(f"Added/updated adaptation rule: {rule.name}")
    
    def save_models(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save the cultural model to disk.
        
        Args:
            output_dir: Directory to save the model (uses model_dir if None)
        """
        save_dir = Path(output_dir) if output_dir else self.model_dir
        if not save_dir:
            raise ValueError("No output directory specified")
            
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save region profiles
        profiles_dir = save_dir / "region_profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        for region_id, profile in self.region_profiles.items():
            profile_path = profiles_dir / f"{region_id}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        
        # Save adaptation rules
        rules_path = save_dir / "adaptation_rules.json"
        rules_data = [
            {
                'name': rule.name,
                'condition': rule.condition,
                'action': rule.action,
                'priority': rule.priority,
                'description': rule.description
            }
            for rule in self.adaptation_rules.values()
        ]
        with open(rules_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        logger.info(f"Saved cultural model to {save_dir}")
    
    def load_models(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load cultural model from disk.
        
        Args:
            model_dir: Directory containing the model files (uses model_dir if None)
        """
        load_dir = Path(model_dir) if model_dir else self.model_dir
        if not load_dir or not load_dir.exists():
            logger.warning(f"Model directory not found: {load_dir}")
            return
            
        # Load region profiles
        profiles_dir = load_dir / "region_profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                    profile = RegionProfile.from_dict(profile_data)
                    self.region_profiles[profile.region_id] = profile
                except Exception as e:
                    logger.error(f"Error loading profile {profile_file}: {e}")
        
        # Load adaptation rules
        rules_path = load_dir / "adaptation_rules.json"
        if rules_path.exists():
            try:
                with open(rules_path, 'r') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = AdaptationRule(
                        name=rule_data['name'],
                        condition=rule_data['condition'],
                        action=rule_data['action'],
                        priority=rule_data.get('priority', 0),
                        description=rule_data.get('description', '')
                    )
                    self.adaptation_rules[rule.name] = rule
            except Exception as e:
                logger.error(f"Error loading adaptation rules: {e}")
        
        logger.info(f"Loaded cultural model from {load_dir}")
    
    def get_cultural_dimensions(self, region_id: str) -> Dict[CulturalDimension, float]:
        """
        Get cultural dimensions for a specific region.
        
        Args:
            region_id: ID of the region
            
        Returns:
            Dictionary of cultural dimensions and their values
        """
        profile = self.region_profiles.get(region_id)
        if not profile:
            logger.warning(f"Region {region_id} not found, using default")
            profile = self.region_profiles[self.default_region]
        return profile.cultural_dimensions
    
    def compute_cultural_distance(self, 
                                culture_a: Union[str, Dict[CulturalDimension, float]], 
                                culture_b: Union[str, Dict[CulturalDimension, float]]) -> float:
        """
        Compute the cultural distance between two cultural profiles.
        
        Args:
            culture_a: First cultural profile (region ID or dict of dimensions)
            culture_b: Second cultural profile (region ID or dict of dimensions)
            
        Returns:
            Cultural distance score (0-1)
        """
        # Get dimension values for culture_a
        if isinstance(culture_a, str):
            dims_a = self.get_cultural_dimensions(culture_a)
        else:
            dims_a = culture_a
            
        # Get dimension values for culture_b
        if isinstance(culture_b, str):
            dims_b = self.get_cultural_dimensions(culture_b)
        else:
            dims_b = culture_b
            
        # Convert to arrays of matching dimensions
        common_dims = set(dims_a.keys()) & set(dims_b.keys())
        if not common_dims:
            logger.warning("No common cultural dimensions found")
            return 1.0  # Maximum distance if no dimensions in common
            
        vec_a = np.array([dims_a[dim] for dim in common_dims])
        vec_b = np.array([dims_b[dim] for dim in common_dims])
        
        # Calculate Euclidean distance and normalize to [0, 1]
        max_possible_distance = np.sqrt(len(common_dims))  # Max distance when all dimensions differ by 1.0
        distance = np.linalg.norm(vec_a - vec_b) / max_possible_distance
        
        return float(np.clip(distance, 0.0, 1.0))
    
    def adapt_response(self, 
                      response: str, 
                      source_culture: Union[str, Dict[CulturalDimension, float]], 
                      target_culture: Union[str, Dict[CulturalDimension, float]],
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Adapt a service response based on cultural differences and context.
        
        Args:
            response: Original service response
            source_culture: Source cultural profile (region ID or dict of dimensions)
            target_culture: Target cultural profile (region ID or dict of dimensions)
            context: Additional context for adaptation
            
        Returns:
            str: Culturally adapted response
        """
        if context is None:
            context = {}
            
        # Get cultural profiles
        if isinstance(source_culture, str):
            source_profile = self.region_profiles.get(source_culture)
            if not source_profile:
                logger.warning(f"Source culture {source_culture} not found, using default")
                source_profile = self.region_profiles[self.default_region]
        else:
            source_profile = RegionProfile("source", "Source Culture", cultural_dimensions=source_culture)
            
        if isinstance(target_culture, str):
            target_profile = self.region_profiles.get(target_culture)
            if not target_profile:
                logger.warning(f"Target culture {target_culture} not found, using default")
                target_profile = self.region_profiles[self.default_region]
        else:
            target_profile = RegionProfile("target", "Target Culture", cultural_dimensions=target_culture)
        
        # Prepare adaptation context
        adaptation_context = {
            'source_culture': source_profile.cultural_dimensions,
            'target_culture': target_profile.cultural_dimensions,
            'response': response,
            'context': context,
            'cultural_distance': self.compute_cultural_distance(
                source_profile.cultural_dimensions,
                target_profile.cultural_dimensions
            )
        }
        
        # Apply adaptation rules
        adapted_response = response
        
        # Get applicable rules (sorted by priority)
        applicable_rules = [
            rule for rule in self.adaptation_rules.values()
            if rule.evaluate(adaptation_context)
        ]
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)
        
        # Apply rules in order of priority
        for rule in applicable_rules:
            try:
                result = rule.apply(adaptation_context)
                if 'response' in result:
                    adapted_response = result['response']
                    adaptation_context['response'] = adapted_response
                    logger.debug(f"Applied adaptation rule: {rule.name}")
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
        
        # Apply default adaptation if no rules matched
        if len(applicable_rules) == 0 and 'default' in self.adaptation_rules:
            try:
                result = self.adaptation_rules['default'].apply(adaptation_context)
                if 'response' in result:
                    adapted_response = result['response']
            except Exception as e:
                logger.error(f"Error applying default adaptation: {e}")
        
        return adapted_response

    def update_with_feedback(self, 
                          feedback: Dict[str, Any], 
                          interaction_data: Dict[str, Any]) -> None:
        """
        Update cultural model based on interaction feedback.
        
        Args:
            feedback: Dictionary containing feedback data including:
                     - score: Feedback score (0-1)
                     - region_id: Target region ID
                     - dimensions_impacted: List of cultural dimensions impacted
            interaction_data: Dictionary containing interaction data
        """
        try:
            score = feedback.get('score', 0.5)  # Default to neutral
            region_id = feedback.get('region_id', self.default_region)
            
            # Get or create region profile
            if region_id not in self.region_profiles:
                self.region_profiles[region_id] = RegionProfile(
                    region_id=region_id,
                    name=f"Learned Profile {region_id}",
                    description=f"Automatically learned profile for {region_id}",
                    cultural_dimensions=self.region_profiles[self.default_region].cultural_dimensions.copy()
                )
            
            profile = self.region_profiles[region_id]
            
            # Update cultural dimensions based on feedback
            learning_rate = 0.05  # How quickly to adapt
            
            # Get the dimensions that were most relevant to this interaction
            dimensions_to_update = feedback.get('dimensions_impacted', [])
            if not dimensions_to_update:
                # If no specific dimensions provided, update all with lower learning rate
                dimensions_to_update = list(CulturalDimension)
                learning_rate *= 0.2
            
            # Apply updates to cultural dimensions
            for dim in dimensions_to_update:
                try:
                    if dim in profile.cultural_dimensions:
                        # Move dimension value towards the direction that would have improved the score
                        current_value = profile.cultural_dimensions[dim]
                        adjustment = (score - 0.5) * learning_rate  # Scale by deviation from neutral
                        new_value = np.clip(current_value + adjustment, 0.0, 1.0)
                        profile.cultural_dimensions[dim] = new_value
                except Exception as e:
                    logger.warning(f"Error updating dimension {dim}: {e}")
            
            # Update adaptation rules based on success/failure
            self._update_adaptation_rules(feedback, interaction_data)
            
            # Save updated models if a model directory is configured
            if self.model_dir:
                self.save_models()
                
        except Exception as e:
            logger.error(f"Error updating cultural model: {e}", exc_info=True)
    
    def _update_adaptation_rules(self, 
                              feedback: Dict[str, Any],
                              interaction_data: Dict[str, Any]) -> None:
        """
        Update adaptation rules based on feedback.
        
        Args:
            feedback: Feedback data
            interaction_data: Interaction data
        """
        # This is a placeholder for rule learning/updating logic
        # In a real implementation, this would use reinforcement learning or
        # other techniques to improve the adaptation rules
        pass
