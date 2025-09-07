import numpy as np
import pandas as pd
from typing import List, Dict, Any
import random
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """
    Generates synthetic service interaction data for the SEDS framework.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Define cultural dimensions (based on extended Hofstede model)
        self.cultural_dimensions = [
            'power_distance', 'individualism', 'masculinity', 'uncertainty_avoidance',
            'long_term_orientation', 'indulgence', 'monochronic_time', 'high_context',
            'universalism', 'specificity', 'achievement', 'affective_expressiveness',
            'instrumentality', 'face_saving', 'collective_responsibility', 'hierarchy',
            'formality', 'directness', 'emotional_expression', 'relationship_focus',
            'time_orientation', 'communication_style', 'decision_making', 'trust_basis',
            'conflict_approach'
        ]
        
        # Common service scenarios
        self.service_scenarios = [
            'product_inquiry', 'complaint', 'technical_support', 'billing_question',
            'product_return', 'account_management', 'general_question', 'feedback'
        ]
        
        # Emotional states with base probabilities
        self.emotions = {
            'happy': 0.3,
            'satisfied': 0.3,
            'neutral': 0.5,
            'frustrated': 0.2,
            'angry': 0.1,
            'confused': 0.15,
            'grateful': 0.25,
            'disappointed': 0.2
        }
        
    def generate_cultural_profile(self, region: str = None) -> Dict[str, float]:
        """
        Generate a synthetic cultural profile.
        
        Args:
            region: Optional region to bias the profile
            
        Returns:
            Dictionary of cultural dimension scores (0-1)
        """
        # Base profile with some randomness
        profile = {dim: np.random.beta(2, 2) for dim in self.cultural_dimensions}
        
        # Apply regional biases if specified
        if region == 'north_america':
            profile['individualism'] = np.random.beta(8, 2)  # High individualism
            profile['directness'] = np.random.beta(8, 2)     # Direct communication
            profile['formality'] = np.random.beta(3, 7)      # Low formality
            
        elif region == 'east_asia':
            profile['collective_responsibility'] = np.random.beta(8, 2)
            profile['face_saving'] = np.random.beta(8, 2)
            profile['high_context'] = np.random.beta(8, 2)
            
        elif region == 'middle_east':
            profile['relationship_focus'] = np.random.beta(8, 2)
            profile['formality'] = np.random.beta(8, 2)
            profile['hierarchy'] = np.random.beta(8, 2)
            
        return profile
    
    def generate_emotion_state(self, scenario: str) -> Dict[str, float]:
        """
        Generate synthetic emotion state based on scenario.
        
        Args:
            scenario: Type of service scenario
            
        Returns:
            Dictionary of emotion scores
        """
        # Base probabilities
        probs = self.emotions.copy()
        
        # Adjust based on scenario
        if scenario == 'complaint':
            probs['frustrated'] *= 2.5
            probs['angry'] *= 2.0
            probs['happy'] *= 0.5
        elif scenario == 'technical_support':
            probs['confused'] *= 1.8
            probs['frustrated'] *= 1.5
        elif scenario == 'feedback':
            probs['happy'] *= 1.5
            probs['grateful'] *= 1.8
            
        # Sample emotions based on adjusted probabilities
        emotions = {}
        for emotion, prob in probs.items():
            if random.random() < prob:
                # Generate intensity (higher variance for more extreme emotions)
                intensity = np.random.beta(2, 5) if emotion in ['angry', 'frustrated'] else np.random.beta(2, 2)
                emotions[emotion] = min(1.0, intensity * 1.5)  # Cap at 1.0
                
        return emotions
    
    def generate_interaction(self, interaction_id: int, regions: List[str] = None) -> Dict[str, Any]:
        """
        Generate a single synthetic service interaction.
        
        Args:
            interaction_id: Unique identifier for the interaction
            regions: Optional list of regions to sample from
            
        Returns:
            Dictionary containing interaction data
        """
        # Randomly select a region if not specified
        region = random.choice(regions) if regions else random.choice([
            'north_america', 'europe', 'east_asia', 'south_asia', 
            'middle_east', 'latin_america', 'africa', 'oceania'
        ])
        
        # Generate cultural profile
        cultural_profile = self.generate_cultural_profile(region)
        
        # Select a service scenario
        scenario = random.choices(
            population=list(self.service_scenarios),
            weights=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05],
            k=1
        )[0]
        
        # Generate emotion state
        emotion_state = self.generate_emotion_state(scenario)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=random.randint(0, 30))
        duration = random.randint(30, 600)  # 30 seconds to 10 minutes
        end_time = start_time + timedelta(seconds=duration)
        
        return {
            'interaction_id': interaction_id,
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'region': region,
            'scenario': scenario,
            'cultural_profile': cultural_profile,
            'emotion_state': emotion_state,
            'satisfaction_score': self._generate_satisfaction_score(scenario, emotion_state),
            'resolution_status': random.choices(
                ['resolved', 'escalated', 'pending', 'follow_up_required'],
                weights=[0.7, 0.1, 0.1, 0.1],
                k=1
            )[0]
        }
    
    def _generate_satisfaction_score(self, scenario: str, emotion_state: Dict[str, float]) -> float:
        """
        Generate a synthetic satisfaction score based on scenario and emotions.
        """
        # Base score
        score = 0.5
        
        # Adjust based on emotions
        for emotion, intensity in emotion_state.items():
            if emotion in ['happy', 'satisfied', 'grateful']:
                score += 0.2 * intensity
            elif emotion in ['frustrated', 'angry', 'disappointed']:
                score -= 0.3 * intensity
                
        # Adjust based on scenario
        if scenario == 'complaint':
            score -= 0.2
        elif scenario == 'feedback':
            score += 0.1
            
        # Add some noise
        score += random.uniform(-0.1, 0.1)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def generate_dataset(self, n_samples: int = 1000, regions: List[str] = None) -> pd.DataFrame:
        """
        Generate a dataset of synthetic service interactions.
        
        Args:
            n_samples: Number of interactions to generate
            regions: Optional list of regions to sample from
            
        Returns:
            DataFrame containing the generated interactions
        """
        interactions = []
        for i in range(n_samples):
            interaction = self.generate_interaction(i, regions)
            interactions.append(interaction)
            
        return pd.DataFrame(interactions)


def save_dataset(df: pd.DataFrame, filepath: str):
    """
    Save the generated dataset to a file.
    
    Args:
        df: DataFrame containing the data
        filepath: Path to save the file (should include .csv or .parquet extension)
    """
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")


if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('../data/raw', exist_ok=True)
    
    # Generate and save the dataset
    print("Generating synthetic service interaction data...")
    generator = SyntheticDataGenerator()
    dataset = generator.generate_dataset(n_samples=1000)
    
    # Save in multiple formats
    save_dataset(dataset, '../data/raw/service_interactions.csv')
    save_dataset(dataset, '../data/raw/service_interactions.parquet')
    
    print(f"Generated dataset with {len(dataset)} interactions")
    print("Sample data:")
    print(dataset.head())
