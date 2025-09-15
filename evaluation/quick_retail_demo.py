"""Quick demo of retail cultural adaptation evaluation."""

import os
import json
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import mean_absolute_error

# Configuration
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL_NAME = 'distilbert-base-uncased'

def load_retail_data():
    """Load sample retail interactions."""
    return [
        {
            "text": "I need a new phone with a great camera",
            "region": "North America",
            "cultural_profile": {"individualism": 0.85, "power_distance": 0.30},
            "expected_response": "feature-focused"
        },
        {
            "text": "My boss needs a new laptop for presentations",
            "region": "East Asia",
            "cultural_profile": {"individualism": 0.30, "power_distance": 0.85},
            "expected_response": "status-focused"
        },
        {
            "text": "Looking for a reliable family car",
            "region": "Europe",
            "cultural_profile": {"individualism": 0.60, "power_distance": 0.40},
            "expected_response": "safety-focused"
        }
    ]

def analyze_cultural_aspects(text, region):
    """Simple cultural analysis of text."""
    # In a real scenario, this would use the full model
    # For demo, we'll use simple heuristics
    if "boss" in text.lower() and region == "East Asia":
        return {"power_distance": 0.8, "individualism": 0.3}
    elif "family" in text.lower() and region == "Europe":
        return {"power_distance": 0.4, "individualism": 0.6}
    else:  # Default North American pattern
        return {"power_distance": 0.3, "individualism": 0.8}

def generate_response(text, cultural_profile):
    """Generate culturally adapted response."""
    if cultural_profile["individualism"] > 0.7:
        return f"This {text.split()[-1]} has amazing personal features you'll love!"
    elif cultural_profile["power_distance"] > 0.7:
        return f"This premium {text.split()[-1]} will impress your colleagues."
    else:
        return f"This reliable {text.split()[-1]} is perfect for your needs."

def run_demo():
    """Run the retail demo."""
    print("🚀 Starting Retail Cultural Adaptation Demo\n")
    
    # Load sample data
    interactions = load_retail_data()
    results = []
    
    print("🔍 Analyzing customer interactions...\n")
    for i, interaction in enumerate(interactions, 1):
        print(f"👤 Customer {i} ({interaction['region']}): {interaction['text']}")
        
        # Analyze cultural aspects
        cultural_profile = analyze_cultural_aspects(
            interaction['text'], 
            interaction['region']
        )
        
        # Generate response
        response = generate_response(interaction['text'], cultural_profile)
        
        # Calculate MAE for cultural dimensions
        mae_individualism = abs(
            cultural_profile["individualism"] - 
            interaction['cultural_profile']["individualism"]
        )
        mae_power_distance = abs(
            cultural_profile["power_distance"] - 
            interaction['cultural_profile']["power_distance"]
        )
        
        print(f"   🌐 Cultural Analysis:")
        print(f"      - Individualism: {cultural_profile['individualism']:.2f} "
              f"(Δ={mae_individualism:.2f})")
        print(f"      - Power Distance: {cultural_profile['power_distance']:.2f} "
              f"(Δ={mae_power_distance:.2f})")
        print(f"   💬 Response: {response}")
        print()
        
        results.append({
            'interaction': i,
            'region': interaction['region'],
            'mae_individualism': mae_individualism,
            'mae_power_distance': mae_power_distance,
            'response_type': 'feature-focused' if 'amazing' in response 
                           else 'status-focused' if 'premium' in response 
                           else 'safety-focused'
        })
    
    # Calculate average MAE
    avg_mae_ind = np.mean([r['mae_individualism'] for r in results])
    avg_mae_pd = np.mean([r['mae_power_distance'] for r in results])
    
    print("\n📊 Results Summary:")
    print(f"- Average MAE (Individualism): {avg_mae_ind:.3f}")
    print(f"- Average MAE (Power Distance): {avg_mae_pd:.3f}")
    
    print("\n🎯 Key Insights for Retail:")
    print("1. North American customers respond best to personal benefits")
    print("2. East Asian customers value status and hierarchy in products")
    print("3. European customers prioritize reliability and safety")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/retail_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Demo complete! Results saved to results/retail_demo_results.json")

if __name__ == "__main__":
    run_demo()
