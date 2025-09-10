"""
Enhanced Cultural Model for SEDS with advanced adaptation and expert validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import random
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CulturalDimension(str, Enum):
    """Extended cultural dimensions with detailed descriptions."""
    # Hofstede's dimensions
    POWER_DISTANCE = "power_distance"  # Acceptance of hierarchical distribution of power
    INDIVIDUALISM = "individualism"   # Degree of interdependence among society members
    MASCULINITY = "masculinity"       # Distribution of emotional roles between genders
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"  # Tolerance for ambiguity
    LONG_TERM_ORIENTATION = "long_term_orientation"  # Time horizon focus
    INDULGENCE = "indulgence"         # Gratification vs. restraint of desires
    
    # Hall's dimensions
    CONTEXT = "context"               # High vs. low context communication
    TIME = "time_orientation"         # Monochronic vs. polychronic time
    SPACE = "space"                   # Personal space and territoriality
    
    # Schwartz's cultural values
    EMBEDDEDNESS = "embeddedness"     # Social order and tradition
    HIERARCHY = "hierarchy"          # Legitimate hierarchical allocation of roles
    MASTERY = "mastery"              # Getting ahead through active self-assertion
    AFFECTIVE_AUTONOMY = "affective_autonomy"  # Pursuit of positive experiences
    INTELLECTUAL_AUTONOMY = "intellectual_autonomy"  # Independent ideas and rights
    EGALITARIANISM = "egalitarianism"  # Transcendence of selfish interests
    HARMONY = "harmony"              # Fitting harmoniously into the environment

class CulturalAdaptationStrategy(Enum):
    """Strategies for cultural adaptation."""
    ASSIMILATION = "assimilation"     # Fully adopt target culture
    INTEGRATION = "integration"       # Balance between cultures
    SEPARATION = "separation"         # Maintain original culture
    MARGINALIZATION = "marginalization"  # Reject both cultures
    HYBRIDIZATION = "hybridization"   # Create new cultural forms

@dataclass
class CulturalProfile:
    """Comprehensive cultural profile with validation and adaptation capabilities."""
    profile_id: str
    name: str
    description: str = ""
    dimensions: Dict[CulturalDimension, float] = field(default_factory=dict)
    adaptation_rules: Dict[str, Dict] = field(default_factory=dict)
    validation_status: str = "unvalidated"
    expert_reviews: List[Dict] = field(default_factory=list)
    
    def validate(self, expert_id: str, feedback: Dict) -> bool:
        """Validate the profile with expert feedback."""
        self.validation_status = feedback.get('status', 'reviewed')
        self.expert_reviews.append({
            'expert_id': expert_id,
            'timestamp': datetime.utcnow().isoformat(),
            'feedback': feedback,
            'status_before': self.validation_status
        })
        return self.validation_status == 'validated'
    
    def calculate_distance(self, other: 'CulturalProfile', 
                          weights: Optional[Dict[CulturalDimension, float]] = None) -> float:
        """Calculate cultural distance using weighted Euclidean distance."""
        if not weights:
            weights = {dim: 1.0 for dim in self.dimensions}
            
        squared_diff = 0.0
        total_weight = 0.0
        
        for dim, weight in weights.items():
            if dim in self.dimensions and dim in other.dimensions:
                diff = self.dimensions[dim] - other.dimensions[dim]
                squared_diff += (weight * diff) ** 2
                total_weight += weight ** 2
                
        return np.sqrt(squared_diff / total_weight) if total_weight > 0 else 0.0

class CulturalAdaptationEngine:
    """Advanced cultural adaptation engine with learning capabilities."""
    
    def __init__(self, dimensions: int = len(CulturalDimension)):
        self.dimensions = dimensions
        self.profiles: Dict[str, CulturalProfile] = {}
        self.adaptation_strategies = {
            'default': self._default_adaptation,
            'contextual': self._contextual_adaptation,
            'reinforcement': self._reinforcement_adaptation,
            'hybrid': self._hybrid_adaptation
        }
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.memory = []
        self.memory_capacity = 1000
        
    def add_profile(self, profile: CulturalProfile) -> None:
        """Add or update a cultural profile."""
        self.profiles[profile.profile_id] = profile
        
    def get_adaptation_plan(self, source_id: str, target_id: str, 
                           context: Optional[Dict] = None) -> Dict:
        """Generate a cultural adaptation plan between two profiles."""
        if source_id not in self.profiles or target_id not in self.profiles:
            raise ValueError("Source or target profile not found")
            
        source = self.profiles[source_id]
        target = self.profiles[target_id]
        
        # Choose adaptation strategy based on context
        strategy = self._select_strategy(context)
        
        # Generate adaptation plan
        plan = {
            'strategy': strategy,
            'dimensions': {},
            'actions': [],
            'expected_impact': 0.0,
            'confidence': 0.0
        }
        
        # Calculate dimensional differences
        for dim in source.dimensions:
            if dim in target.dimensions:
                diff = target.dimensions[dim] - source.dimensions[dim]
                plan['dimensions'][dim] = {
                    'source': source.dimensions[dim],
                    'target': target.dimensions[dim],
                    'difference': diff,
                    'adaptation': self._calculate_adaptation(dim, diff, strategy, context)
                }
        
        # Add adaptation actions
        plan['actions'] = self._generate_actions(plan['dimensions'], context)
        
        return plan
    
    def _select_strategy(self, context: Optional[Dict]) -> str:
        """Select the most appropriate adaptation strategy."""
        if not context:
            return 'default'
            
        # Simple strategy selection - can be enhanced with ML
        if random.random() < self.exploration_rate:
            return random.choice(list(self.adaptation_strategies.keys()))
            
        # Use contextual information to select strategy
        if context.get('urgency', 0) > 0.7:
            return 'contextual'
        elif context.get('learning_enabled', False):
            return 'reinforcement'
        else:
            return 'hybrid'
    
    def _calculate_adaptation(self, dimension: CulturalDimension, difference: float,
                            strategy: str, context: Optional[Dict]) -> float:
        """Calculate the adaptation amount for a dimension."""
        strategy_fn = self.adaptation_strategies.get(strategy, self._default_adaptation)
        return strategy_fn(dimension, difference, context)
    
    def _default_adaptation(self, dimension: CulturalDimension, difference: float,
                           context: Optional[Dict]) -> float:
        """Default linear adaptation."""
        return difference * self.learning_rate
    
    def _contextual_adaptation(self, dimension: CulturalDimension, difference: float,
                              context: Optional[Dict]) -> float:
        """Context-aware adaptation considering environmental factors."""
        base_adaptation = self._default_adaptation(dimension, difference, context)
        
        # Adjust based on context
        if context:
            urgency = context.get('urgency', 0.5)
            importance = context.get('importance', {}).get(dimension, 0.5)
            return base_adaptation * (1 + urgency) * (0.5 + importance/2)
        return base_adaptation
    
    def _reinforcement_adaptation(self, dimension: CulturalDimension, difference: float,
                                context: Optional[Dict]) -> float:
        """Reinforcement learning-based adaptation."""
        # Placeholder for RL-based adaptation
        # In practice, this would use a learned policy
        return self._default_adaptation(dimension, difference, context) * (0.8 + 0.4 * random.random())
    
    def _hybrid_adaptation(self, dimension: CulturalDimension, difference: float,
                          context: Optional[Dict]) -> float:
        """Hybrid adaptation combining multiple strategies."""
        strategies = [
            self._default_adaptation,
            self._contextual_adaptation,
            self._reinforcement_adaptation
        ]
        weights = [0.4, 0.3, 0.3]  # Can be learned
        
        adaptations = [s(dimension, difference, context) for s in strategies]
        return sum(w * a for w, a in zip(weights, adaptations))
    
    def _generate_actions(self, dimensions: Dict, context: Optional[Dict]) -> List[Dict]:
        """Generate concrete adaptation actions based on dimensional analysis."""
        actions = []
        
        for dim, data in dimensions.items():
            diff = data['difference']
            if abs(diff) < 0.1:  # Skip small differences
                continue
                
            action = {
                'dimension': dim,
                'current_value': data['source'],
                'target_value': data['target'],
                'adaptation': data['adaptation'],
                'suggestions': self._get_suggestions(dim, diff, context)
            }
            actions.append(action)
            
        return sorted(actions, key=lambda x: abs(x['adaptation']), reverse=True)
    
    def _get_suggestions(self, dimension: CulturalDimension, difference: float,
                        context: Optional[Dict]) -> List[str]:
        """Generate human-readable adaptation suggestions."""
        suggestions = []
        dim_name = dimension.value.replace('_', ' ').title()
        
        if abs(difference) > 0.3:  # Significant difference
            direction = "increase" if difference > 0 else "decrease"
            suggestions.append(f"Significantly {direction} {dim_name} in communications")
        elif abs(difference) > 0.1:  # Moderate difference
            direction = "slightly increase" if difference > 0 else "slightly decrease"
            suggestions.append(f"Consider to {direction} {dim_name} in interactions")
            
        # Add context-specific suggestions
        if context and context.get('domain') == 'business':
            if dimension == CulturalDimension.POWER_DISTANCE and difference > 0:
                suggestions.append("Use more formal titles and honorifics in business communications")
            elif dimension == CulturalDimension.INDIVIDUALISM and difference < 0:
                suggestions.append("Emphasize team achievements over individual accomplishments")
                
        return suggestions
    
    def learn_from_feedback(self, source_id: str, target_id: str, 
                           feedback: Dict, context: Optional[Dict] = None) -> None:
        """Update the adaptation model based on feedback."""
        # Store experience in memory
        experience = {
            'source_id': source_id,
            'target_id': target_id,
            'feedback': feedback,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.memory.append(experience)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)  # Remove oldest experience
            
        # Update learning parameters (simplified)
        if feedback.get('success', False):
            # Reinforce successful adaptations
            self.learning_rate = min(0.5, self.learning_rate * 1.05)
        else:
            # Reduce learning rate on failure
            self.learning_rate = max(0.01, self.learning_rate * 0.95)

class ExpertValidationSystem:
    """System for managing expert validation of cultural profiles and adaptations."""
    
    def __init__(self):
        self.validators = {}
        self.validation_queue = []
        self.validated_profiles = {}
        
    def register_validator(self, expert_id: str, domains: List[str], 
                         expertise_level: int = 3) -> bool:
        """Register a domain expert for validation."
        
        Args:
            expert_id: Unique identifier for the expert
            domains: List of cultural domains the expert can validate
            expertise_level: Self-reported expertise level (1-5)
            
        Returns:
            bool: True if registration was successful
        """
        self.validators[expert_id] = {
            'domains': domains,
            'expertise_level': max(1, min(5, expertise_level)),
            'validations_completed': 0,
            'validation_score': 0.0
        }
        return True
    
    def submit_for_validation(self, profile: CulturalProfile, 
                            priority: str = 'normal') -> str:
        """Submit a cultural profile for expert validation.
        
        Args:
            profile: CulturalProfile to validate
            priority: Validation priority ('low', 'normal', 'high')
            
        Returns:
            str: Validation request ID
        """
        request_id = f"val_{len(self.validation_queue)}_{hash(profile.profile_id)}"
        
        self.validation_queue.append({
            'request_id': request_id,
            'profile': profile,
            'priority': priority,
            'status': 'pending',
            'submitted_at': datetime.utcnow(),
            'assigned_to': None
        })
        
        return request_id
    
    def assign_validation(self, expert_id: str) -> Optional[Dict]:
        """Assign a validation task to an expert.
        
        Args:
            expert_id: ID of the expert to assign the task to
            
        Returns:
            Optional[Dict]: Validation task details or None if no tasks available
        """
        if expert_id not in self.validators:
            return None
            
        # Get expert's domains
        expert_domains = set(self.validators[expert_id]['domains'])
        
        # Find highest priority task matching expert's domains
        for task in sorted(self.validation_queue, 
                          key=lambda x: (x['priority'] != 'high', 
                                       x['priority'] != 'normal',
                                       x['submitted_at'])):
            if task['status'] == 'pending':
                # Check if expert has matching domain
                profile_domains = set(task['profile'].dimensions.keys())
                if expert_domains.intersection(profile_domains):
                    task['status'] = 'in_progress'
                    task['assigned_to'] = expert_id
                    task['assigned_at'] = datetime.utcnow()
                    return task
                    
        return None
    
    def submit_validation(self, expert_id: str, request_id: str, 
                         feedback: Dict) -> bool:
        """Submit expert validation feedback.
        
        Args:
            expert_id: ID of the expert submitting feedback
            request_id: Validation request ID
            feedback: Dictionary containing validation results
            
        Returns:
            bool: True if submission was successful
        """
        # Find the validation task
        task = next((t for t in self.validation_queue 
                    if t['request_id'] == request_id and 
                    t['assigned_to'] == expert_id), None)
                    
        if not task:
            return False
            
        # Update task status
        task['status'] = 'completed'
        task['completed_at'] = datetime.utcnow()
        task['feedback'] = feedback
        
        # Update expert stats
        if expert_id in self.validators:
            self.validators[expert_id]['validations_completed'] += 1
            # Simple scoring - can be enhanced
            self.validators[expert_id]['validation_score'] = min(
                5.0,
                self.validators[expert_id]['validation_score'] + 0.1
            )
        
        # Update profile validation status
        profile = task['profile']
        profile.validate(expert_id, feedback)
        
        # Move to validated profiles
        self.validated_profiles[profile.profile_id] = {
            'profile': profile,
            'validation_data': {
                'expert_id': expert_id,
                'timestamp': task['completed_at'],
                'feedback': feedback
            }
        }
        
        return True

# Example usage
if __name__ == "__main__":
    # Create cultural profiles
    us_profile = CulturalProfile(
        profile_id="us_en",
        name="United States (English)",
        description="Cultural profile for English-speaking United States",
        dimensions={
            CulturalDimension.INDIVIDUALISM: 0.9,
            CulturalDimension.POWER_DISTANCE: 0.4,
            CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.46,
            CulturalDimension.MASCULINITY: 0.62,
            CulturalDimension.LONG_TERM_ORIENTATION: 0.26,
            CulturalDimension.INDULGENCE: 0.68
        }
    )
    
    jp_profile = CulturalProfile(
        profile_id="jp_ja",
        name="Japan (Japanese)",
        description="Cultural profile for Japan",
        dimensions={
            CulturalDimension.INDIVIDUALISM: 0.18,
            CulturalDimension.POWER_DISTANCE: 0.54,
            CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.92,
            CulturalDimension.MASCULINITY: 0.95,
            CulturalDimension.LONG_TERM_ORIENTATION: 0.88,
            CulturalDimension.INDULGENCE: 0.42
        }
    )
    
    # Initialize adaptation engine
    engine = CulturalAdaptationEngine()
    engine.add_profile(us_profile)
    engine.add_profile(jp_profile)
    
    # Generate adaptation plan
    context = {
        'domain': 'business',
        'urgency': 0.7,
        'importance': {
            CulturalDimension.POWER_DISTANCE: 0.9,
            CulturalDimension.INDIVIDUALISM: 0.8
        }
    }
    
    plan = engine.get_adaptation_plan("us_en", "jp_ja", context)
    print("\n=== Cultural Adaptation Plan ===")
    print(f"Strategy: {plan['strategy']}")
    
    print("\nDimensional Adaptations:")
    for dim, data in plan['dimensions'].items():
        print(f"- {dim.value}: {data['source']:.2f} -> {data['target']:.2f} "
              f"(adapt: {data['adaptation']:+.2f})")
    
    print("\nSuggested Actions:")
    for i, action in enumerate(plan['actions'], 1):
        print(f"{i}. {action['suggestions'][0]}")
    
    # Example of expert validation
    print("\n=== Expert Validation ===")
    validation_system = ExpertValidationSystem()
    validation_system.register_validator("exp1", ["business", "communication"], 4)
    
    # Submit profile for validation
    request_id = validation_system.submit_for_validation(us_profile, "high")
    print(f"Submitted for validation. Request ID: {request_id}")
    
    # Expert completes validation
    task = validation_system.assign_validation("exp1")
    if task:
        print(f"\nExpert assigned to validate: {task['profile'].name}")
        
        # Submit validation feedback
        feedback = {
            'status': 'validated',
            'confidence': 0.9,
            'notes': "Profile accurately represents US cultural dimensions",
            'suggested_changes': {}
        }
        
        if validation_system.submit_validation("exp1", request_id, feedback):
            print("Validation submitted successfully!")
            print(f"Updated profile status: {us_profile.validation_status}")
