"""Retail dataset for cultural adaptation evaluation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Sample retail interaction data with cultural annotations
RETAIL_INTERACTIONS = [
    {
        "text": "I'm looking for a new smartphone with a good camera.",
        "cultural_profile": {
            "individualism": 0.85,  # High individualism
            "power_distance": 0.30,  # Low power distance
            "uncertainty_avoidance": 0.60,
            "masculinity": 0.45,
            "long_term_orientation": 0.55,
            "indulgence": 0.70
        },
        "emotions": {
            "happy": 0.7,
            "sad": 0.1,
            "angry": 0.1,
            "surprised": 0.1,
            "neutral": 0.0
        },
        "region": "North America",
        "product_category": "Electronics"
    },
    {
        "text": "This shirt is too expensive. Can you give me a discount?",
        "cultural_profile": {
            "individualism": 0.45,
            "power_distance": 0.75,  # High power distance
            "uncertainty_avoidance": 0.80,  # High uncertainty avoidance
            "masculinity": 0.65,
            "long_term_orientation": 0.70,
            "indulgence": 0.30
        },
        "emotions": {
            "happy": 0.1,
            "sad": 0.2,
            "angry": 0.6,
            "surprised": 0.1,
            "neutral": 0.0
        },
        "region": "East Asia",
        "product_category": "Apparel"
    },
    {
        "text": "I need help finding a gift for my wife's birthday.",
        "cultural_profile": {
            "individualism": 0.70,
            "power_distance": 0.40,
            "uncertainty_avoidance": 0.50,
            "masculinity": 0.30,  # More feminine values
            "long_term_orientation": 0.60,
            "indulgence": 0.65
        },
        "emotions": {
            "happy": 0.8,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.1,
            "neutral": 0.1
        },
        "region": "Europe",
        "product_category": "Gifts"
    },
    {
        "text": "The product I received is damaged. I want a refund immediately!",
        "cultural_profile": {
            "individualism": 0.90,  # Very high individualism
            "power_distance": 0.20,  # Very low power distance
            "uncertainty_avoidance": 0.30,
            "masculinity": 0.80,  # High masculinity
            "long_term_orientation": 0.40,
            "indulgence": 0.85  # High indulgence
        },
        "emotions": {
            "happy": 0.0,
            "sad": 0.1,
            "angry": 0.9,
            "surprised": 0.0,
            "neutral": 0.0
        },
        "region": "North America",
        "product_category": "Customer Service"
    },
    {
        "text": "Could you please suggest some traditional gifts for my business partners?",
        "cultural_profile": {
            "individualism": 0.30,  # Low individualism
            "power_distance": 0.85,  # High power distance
            "uncertainty_avoidance": 0.75,
            "masculinity": 0.65,
            "long_term_orientation": 0.80,  # Long-term orientation
            "indulgence": 0.25  # Low indulgence
        },
        "emotions": {
            "happy": 0.3,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.1,
            "neutral": 0.6
        },
        "region": "East Asia",
        "product_category": "Gifts"
    }
]

def generate_retail_dataset(size: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic retail dataset for evaluation.
    
    Args:
        size: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic retail interactions
    """
    np.random.seed(seed)
    
    # Define regions and their cultural profiles
    regions = {
        "North America": {
            "individualism": (0.8, 0.1),  # mean, std
            "power_distance": (0.4, 0.1),
            "uncertainty_avoidance": (0.5, 0.15),
            "masculinity": (0.6, 0.1),
            "long_term_orientation": (0.45, 0.1),
            "indulgence": (0.7, 0.1)
        },
        "Europe": {
            "individualism": (0.7, 0.15),
            "power_distance": (0.35, 0.1),
            "uncertainty_avoidance": (0.65, 0.15),
            "masculinity": (0.5, 0.1),
            "long_term_orientation": (0.6, 0.1),
            "indulgence": (0.6, 0.15)
        },
        "East Asia": {
            "individualism": (0.3, 0.1),
            "power_distance": (0.7, 0.15),
            "uncertainty_avoidance": (0.75, 0.1),
            "masculinity": (0.6, 0.1),
            "long_term_orientation": (0.8, 0.1),
            "indulgence": (0.35, 0.1)
        },
        "Middle East": {
            "individualism": (0.4, 0.15),
            "power_distance": (0.8, 0.1),
            "uncertainty_avoidance": (0.65, 0.15),
            "masculinity": (0.7, 0.1),
            "long_term_orientation": (0.5, 0.15),
            "indulgence": (0.4, 0.1)
        },
        "Latin America": {
            "individualism": (0.35, 0.1),
            "power_distance": (0.7, 0.1),
            "uncertainty_avoidance": (0.8, 0.1),
            "masculinity": (0.6, 0.15),
            "long_term_orientation": (0.4, 0.1),
            "indulgence": (0.7, 0.1)
        }
    }
    
    # Product categories and their emotional profiles
    product_categories = ["Electronics", "Apparel", "Home", "Beauty", "Toys", "Gifts", "Customer Service"]
    
    # Generate synthetic data
    data = []
    for _ in range(size):
        # Sample a region
        region = np.random.choice(list(regions.keys()))
        region_profile = regions[region]
        
        # Generate cultural profile
        cultural_profile = {
            dim: float(np.clip(np.random.normal(mean_std[0], mean_std[1]), 0, 1))
            for dim, mean_std in region_profile.items()
        }
        
        # Sample product category
        product_category = np.random.choice(product_categories)
        
        # Generate emotions based on product category
        base_emotions = {
            "happy": 0.3,
            "sad": 0.1,
            "angry": 0.1,
            "surprised": 0.1,
            "neutral": 0.4
        }
        
        # Adjust emotions based on category
        if product_category == "Customer Service":
            base_emotions.update({
                "happy": 0.1,
                "sad": 0.2,
                "angry": 0.5,
                "surprised": 0.1,
                "neutral": 0.1
            })
        elif product_category == "Gifts":
            base_emotions.update({
                "happy": 0.6,
                "sad": 0.05,
                "angry": 0.05,
                "surprised": 0.2,
                "neutral": 0.1
            })
        
        # Add some noise
        emotions = {
            k: float(np.clip(v + np.random.normal(0, 0.1), 0, 1))
            for k, v in base_emotions.items()
        }
        # Normalize
        total = sum(emotions.values())
        emotions = {k: float(v/total) for k, v in emotions.items()}
        
        # Generate text based on category and emotions
        if product_category == "Electronics":
            texts = [
                f"I'm looking for a new {np.random.choice(['smartphone', 'laptop', 'tablet', 'smartwatch'])} with good {np.random.choice(['battery life', 'camera', 'performance'])}.",
                f"Can you recommend a {np.random.choice(['gaming', 'business', 'student'])} {np.random.choice(['laptop', 'tablet'])}?",
                f"What's the difference between these two {np.random.choice(['smartphones', 'laptops', 'headphones'])}?"
            ]
        elif product_category == "Apparel":
            texts = [
                f"I need a {np.random.choice(['casual', 'formal', 'business'])} {np.random.choice(['shirt', 'dress', 'pants', 'jacket'])} for a {np.random.choice(['wedding', 'business meeting', 'date', 'party'])}.",
                f"Do you have this in a different {np.random.choice(['size', 'color', 'style'])}?",
                f"How does this {np.random.choice(['shirt', 'dress', 'pants'])} fit?"
            ]
        elif product_category == "Customer Service":
            texts = [
                f"The {np.random.choice(['item', 'product'])} I received is {np.random.choice(['damaged', 'not as described', 'missing parts'])}.",
                f"I want to {np.random.choice(['return', 'exchange', 'refund'])} my order.",
                f"I haven't received my order yet. It's been {np.random.choice(['a week', 'two weeks', 'over a month'])}."
            ]
        else:
            texts = [
                f"Tell me more about this {product_category.lower()} product.",
                f"What are the best {product_category.lower()} items you have?",
                f"I'm looking for a {product_category.lower()} {np.random.choice(['gift', 'present'])} for my {np.random.choice(['mother', 'father', 'sister', 'brother', 'friend'])}."
            ]
        
        text = np.random.choice(texts)
        
        data.append({
            "text": text,
            "cultural_profile": cultural_profile,
            "emotions": emotions,
            "region": region,
            "product_category": product_category
        })
    
    # Add real examples
    data.extend(RETAIL_INTERACTIONS)
    
    return pd.DataFrame(data)
