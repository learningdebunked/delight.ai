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
    Advanced Cultural Adaptation Model for the SEDS framework.
    
    Implements:
    - Multi-dimensional cultural space modeling
    - Dynamic adaptation based on cultural distance metrics
    - Context-aware cultural rule application
    - Learning from feedback
    """
    
    class AdaptationStrategy(Enum):
        """Strategies for cultural adaptation."""
        NEUTRAL = "neutral"         # No adaptation
        MINIMAL = "minimal"         # Light adaptation (e.g., politeness markers)
        MODERATE = "moderate"       # Balance between source and target
        STRONG = "strong"           # Strong adaptation to target culture
        CONTEXTUAL = "contextual"   # Adaptation based on context
    
    def __init__(self, dimensions: int = 25, model_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the advanced cultural model.
        
        Args:
            dimensions: Number of cultural dimensions to model
            model_dir: Directory to save/load model files (optional)
        """
        self.dimensions = dimensions
        self.model_dir = Path(model_dir) if model_dir else None
        self.region_profiles: Dict[str, RegionProfile] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.default_region = "global"
        
        # Cultural distance metrics weights
        self.distance_weights = np.ones(dimensions) / dimensions
        
        # Cultural dimension importance (learned)
        self.dimension_importance = np.ones(dimensions) / dimensions
        
        # Adaptation strategy parameters
        self.adaptation_strategy = self.AdaptationStrategy.CONTEXTUAL
        self.strategy_weights = {
            'directness': 0.7,
            'formality': 0.8,
            'hierarchy': 0.6,
            'time_orientation': 0.5
        }
        
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
                      source_culture: Union[str, Dict[str, float]], 
                      target_culture: Union[str, Dict[str, float]],
                      context: Optional[Dict[str, Any]] = None,
                      adaptation_strength: float = 0.5) -> str:
        """
        Adapt a response based on cultural differences using advanced techniques.
        
        Args:
            response: Original response text
            source_culture: Source cultural profile (region ID or dict of dimensions)
            target_culture: Target cultural profile (region ID or dict of dimensions)
            context: Additional context for adaptation
            adaptation_strength: Strength of adaptation (0.0 to 1.0)
            
        Returns:
            Culturally adapted response with metadata
        """
        # Get cultural profiles
        source_profile = self._get_cultural_profile(source_culture)
        target_profile = self._get_cultural_profile(target_culture)
        
        # Calculate cultural distance and adaptation parameters
        distance = self.calculate_cultural_distance(source_profile, target_profile)
        adaptation_params = self._calculate_adaptation_parameters(
            source_profile, 
            target_profile, 
            distance,
            adaptation_strength
        )
        
        # Apply cultural adaptation pipeline
        adapted_response = self._apply_cultural_pipeline(
            response,
            source_profile,
            target_profile,
            adaptation_params,
            context or {}
        )
        
        return adapted_response
        
    def calculate_cultural_distance(self, 
                                  profile_a: Dict[str, float], 
                                  profile_b: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate multiple cultural distance metrics between two profiles.
        
        Args:
            profile_a: First cultural profile
            profile_b: Second cultural profile
            
        Returns:
            Dictionary of distance metrics and their values
        """
        # Convert profiles to arrays for numerical operations
        a = np.array([profile_a.get(dim, 0.5) for dim in CulturalDimension])
        b = np.array([profile_b.get(dim, 0.5) for dim in CulturalDimension])
        
        # Calculate multiple distance metrics
        euclidean = np.linalg.norm(a - b)
        manhattan = np.sum(np.abs(a - b))
        cosine = 1 - cosine_similarity([a], [b])[0][0]
        
        # Calculate weighted Mahalanobis distance (requires covariance matrix)
        # Using identity matrix as default covariance
        cov_inv = np.eye(len(a))  # In practice, this should be learned from data
        mahalanobis_dist = mahalanobis(a, b, cov_inv)
        
        # Calculate Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = wasserstein_distance(a, b)
        
        return {
            'euclidean': float(euclidean),
            'manhattan': float(manhattan),
            'cosine': float(cosine),
            'mahalanobis': float(mahalanobis_dist),
            'wasserstein': float(wasserstein_dist),
            'weighted': float(np.sum(self.distance_weights * np.abs(a - b)))
        }
    
    def _calculate_adaptation_parameters(self,
                                       source_profile: Dict[str, float],
                                       target_profile: Dict[str, float],
                                       distance_metrics: Dict[str, float],
                                       strength: float = 0.5) -> Dict[str, Any]:
        """
        Calculate parameters for cultural adaptation.
        
        Args:
            source_profile: Source cultural profile
            target_profile: Target cultural profile
            distance_metrics: Dictionary of distance metrics
            strength: Overall adaptation strength (0.0 to 1.0)
            
        Returns:
            Dictionary of adaptation parameters
        """
        # Calculate dimension-specific adaptation strengths
        dim_adaptation = {}
        for dim in CulturalDimension:
            dim_diff = abs(source_profile.get(dim, 0.5) - target_profile.get(dim, 0.5))
            dim_adaptation[dim] = min(1.0, dim_diff * strength * 2)
        
        # Determine adaptation strategy
        if self.adaptation_strategy == self.AdaptationStrategy.CONTEXTUAL:
            # Contextual strategy adapts different aspects differently
            strategy = {
                'directness': self._get_adaptation_level(
                    source_profile.get(CulturalDimension.DIRECTNESS, 0.5),
                    target_profile.get(CulturalDimension.DIRECTNESS, 0.5),
                    strength
                ),
                'formality': self._get_adaptation_level(
                    source_profile.get(CulturalDimension.FORMALITY, 0.5),
                    target_profile.get(CulturalDimension.FORMALITY, 0.5),
                    strength
                ),
                'hierarchy': self._get_adaptation_level(
                    source_profile.get(CulturalDimension.HIERARCHY, 0.5),
                    target_profile.get(CulturalDimension.HIERARCHY, 0.5),
                    strength
                )
            }
        else:
            # Use the same adaptation level for all aspects
            level = strength
            if self.adaptation_strategy == self.AdaptationStrategy.MINIMAL:
                level *= 0.3
            elif self.adaptation_strategy == self.AdaptationStrategy.MODERATE:
                level *= 0.6
            elif self.adaptation_strategy == self.AdaptationStrategy.STRONG:
                level *= 1.0
            else:  # NEUTRAL
                level = 0.0
            
            strategy = {k: level for k in self.strategy_weights.keys()}
        
        return {
            'dimension_adaptation': dim_adaptation,
            'strategy': strategy,
            'overall_strength': strength,
            'distance_metrics': distance_metrics
        }
    
    def _get_adaptation_level(self, source: float, target: float, strength: float) -> float:
        """Calculate adaptation level based on cultural dimension difference."""
        diff = abs(source - target)
        return min(1.0, diff * strength * 2)
    
    def _apply_cultural_pipeline(self,
                               text: str,
                               source_profile: Dict[str, float],
                               target_profile: Dict[str, float],
                               params: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the full cultural adaptation pipeline to the text.
        
        Returns:
            Dictionary containing:
            - adapted_text: The culturally adapted text
            - metadata: Information about the adaptation process
        """
        # Start with the original text
        adapted = text
        
        # Track changes for metadata
        changes = []
        
        # 1. Apply directness adaptation
        if params['strategy']['directness'] > 0.1:  # Only if significant adaptation needed
            directness_diff = (target_profile.get(CulturalDimension.DIRECTNESS, 0.5) - 
                             source_profile.get(CulturalDimension.DIRECTNESS, 0.5))
            adapted, directness_changes = self._adapt_directness(
                adapted, 
                directness_diff * params['strategy']['directness']
            )
            changes.extend(directness_changes)
        
        # 2. Apply formality adaptation
        if params['strategy']['formality'] > 0.1:
            formality_diff = (target_profile.get(CulturalDimension.FORMALITY, 0.5) - 
                            source_profile.get(CulturalDimension.FORMALITY, 0.5))
            adapted, formality_changes = self._adapt_formality(
                adapted,
                formality_diff * params['strategy']['formality']
            )
            changes.extend(formality_changes)
        
        # 3. Apply hierarchical adaptation
        if params['strategy']['hierarchy'] > 0.1:
            hierarchy_diff = (target_profile.get(CulturalDimension.HIERARCHY, 0.5) - 
                            source_profile.get(CulturalDimension.HIERARCHY, 0.5))
            adapted, hierarchy_changes = self._adapt_hierarchy(
                adapted,
                hierarchy_diff * params['strategy']['hierarchy'],
                context
            )
            changes.extend(hierarchy_changes)
        
        # 4. Apply culture-specific rules
        adapted, rule_changes = self._apply_cultural_rules(adapted, context)
        changes.extend(rule_changes)
        
        # Prepare metadata
        metadata = {
            'original_text': text,
            'adapted_text': adapted,
            'source_culture': source_profile,
            'target_culture': target_profile,
            'adaptation_parameters': params,
            'changes': changes,
            'adaptation_strategy': self.adaptation_strategy.value,
            'success': adapted != text  # Whether any adaptation was applied
        }
        
        return {
            'text': adapted,
            'metadata': metadata
        }
    
    def update_with_feedback(self, 
                          feedback: Dict[str, Any], 
                          interaction_data: Dict[str, Any]):
        """
        Update cultural model based on interaction feedback using online learning.
        
        Args:
            feedback: Dictionary containing feedback data including:
                     - score: Feedback score (0-1)
                     - region_id: Target region ID
                     - dimensions_impacted: List of cultural dimensions impacted
                     - adaptation_parameters: Parameters used for adaptation
            interaction_data: Dictionary containing interaction data
        """
        try:
            # Extract feedback data
            score = feedback.get('score', 0.5)  # Default to neutral if not provided
            region_id = feedback.get('region_id', self.default_region)
            
            # Update adaptation rules with reinforcement learning
            self._update_adaptation_rules_with_rl(feedback, interaction_data)
            
            # Update cultural profiles with online learning
            if region_id in self.region_profiles:
                self._update_region_profile_with_learning(
                    region_id, 
                    feedback, 
                    interaction_data
                )
            
            # Update distance metric weights based on feedback
            self._update_distance_metrics(feedback)
            
            # Update adaptation strategy if needed
            self._update_adaptation_strategy(feedback)
            
            logger.info(f"Updated cultural model with feedback score: {score:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating cultural model: {e}", exc_info=True)
    
    def _update_distance_metrics(self, feedback: Dict[str, Any]) -> None:
        """Update distance metric weights based on feedback."""
        if 'distance_metrics' not in feedback or 'score' not in feedback:
            return
            
        score = feedback['score']  # Higher is better
        metrics = feedback['distance_metrics']
        
        # Simple reinforcement learning: increase weight for metrics that led to good outcomes
        learning_rate = 0.1
        for metric, weight in self.distance_weights.items():
            if metric in metrics:
                # If this metric was important and we got good feedback, increase its weight
                if score > 0.7:  # Strong positive feedback
                    self.distance_weights[metric] *= (1 + learning_rate)
                elif score < 0.3:  # Strong negative feedback
                    self.distance_weights[metric] *= (1 - learning_rate)
        
        # Renormalize weights
        total = sum(self.distance_weights.values())
        if total > 0:
            self.distance_weights = {k: v/total for k, v in self.distance_weights.items()}
    
    def _update_adaptation_strategy(self, feedback: Dict[str, Any]) -> None:
        """Update adaptation strategy based on feedback."""
        if 'strategy_performance' not in feedback:
            return
            
        # Update strategy weights based on performance
        strategy_perf = feedback['strategy_performance']
        for strategy, perf in strategy_perf.items():
            if strategy in self.strategy_weights:
                # Simple reinforcement learning update
                self.strategy_weights[strategy] = (
                    0.9 * self.strategy_weights[strategy] + 
                    0.1 * perf
                )
        
        # Optionally switch to a different strategy if one is performing much better
        best_strategy = max(self.strategy_weights, key=self.strategy_weights.get)
        if self.strategy_weights[best_strategy] > (
            self.strategy_weights.get(self.adaptation_strategy.value, 0) + 0.2
        ):
            self.adaptation_strategy = self.AdaptationStrategy(best_strategy)
            logger.info(f"Switched to {best_strategy} adaptation strategy")
    
    def _update_adaptation_rules_with_rl(self, 
                                      feedback: Dict[str, Any],
                                      interaction_data: Dict[str, Any]) -> None:
        """
        Update adaptation rules using reinforcement learning.
        
        Args:
            feedback: Dictionary containing feedback and reward signal
            interaction_data: Context and state information
        """
        try:
            # Extract reward from feedback (normalized to [-1, 1])
            reward = feedback.get('score', 0.0) * 2 - 1  # Convert [0,1] to [-1,1]
            
            # Get the adaptation parameters used
            adaptation_params = feedback.get('adaptation_parameters', {})
            
            # Update rule weights based on performance
            for rule_id, rule in self.adaptation_rules.items():
                if rule.was_applied(interaction_data):
                    # Update rule weight using policy gradient
                    # Simple implementation: increase weight if reward is positive, decrease if negative
                    learning_rate = 0.01
                    rule.weight += learning_rate * reward * rule.weight
                    
                    # Ensure weight stays in reasonable bounds
                    rule.weight = max(0.1, min(1.0, rule.weight))
                    
                    logger.debug(f"Updated rule {rule_id} weight to {rule.weight:.3f} "
                               f"with reward {reward:.2f}")
            
            # Optionally learn new rules from successful adaptations
            if reward > 0.7:  # Strong positive feedback
                self._learn_new_rule(interaction_data, adaptation_params)
                
        except Exception as e:
            logger.error(f"Error updating adaptation rules: {e}", exc_info=True)
    
    def _learn_new_rule(self, 
                       interaction_data: Dict[str, Any],
                       adaptation_params: Dict[str, Any]) -> None:
        """
        Learn a new adaptation rule from a successful interaction.
        
        Args:
            interaction_data: Context and state of the interaction
            adaptation_params: Parameters that led to success
        """
        try:
            # Extract features that might be relevant for a new rule
            context = interaction_data.get('context', {})
            cultural_dims = context.get('cultural_dimensions', {})
            
            # Simple heuristic: create a rule if we see a strong pattern
            # This is a placeholder - in practice, you'd use more sophisticated ML
            if 'directness' in cultural_dims and cultural_dims['directness'] > 0.7:
                # Example: If high directness led to success, create a directness rule
                rule_id = f"directness_rule_{len(self.adaptation_rules)}"
                condition = "context.get('cultural_dimensions', {}).get('directness', 0) > 0.7"
                action = """
                # Make the message more direct
                if '?' in text and not text.strip().endswith('?'):
                    # Move question to the front if it's buried
                    parts = [p for p in text.split('?') if p.strip()]
                    if len(parts) > 1:
                        return f"{parts[-1].strip()}? {' '.join(parts[:-1])}"
                return text
                """
                
                new_rule = AdaptationRule(
                    name=rule_id,
                    condition=condition,
                    action=action,
                    description="Makes messages more direct by moving questions to the front"
                )
                
                self.adaptation_rules[rule_id] = new_rule
                logger.info(f"Learned new adaptation rule: {rule_id}")
                
        except Exception as e:
            logger.error(f"Error learning new rule: {e}", exc_info=True)
