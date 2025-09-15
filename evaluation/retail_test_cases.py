"""Retail test cases for Delight.AI SEDS validation."""

RETAIL_TEST_CASES = [
    # Customer Service Scenarios
    {
        "scenario": "High-value customer complaint in high power distance culture",
        "customer_text": "I am a loyal customer for 10 years. This product is unacceptable.",
        "region": "East Asia",
        "expected_response": {
            "apology_strength": 0.9,
            "compensation_offered": True,
            "formality_level": 0.8
        },
        "cultural_profile": {
            "individualism": 0.3,
            "power_distance": 0.9,
            "uncertainty_avoidance": 0.8,
            "masculinity": 0.6,
            "long_term_orientation": 0.7,
            "indulgence": 0.4
        }
    },
    
    # Product Recommendation Scenarios
    {
        "scenario": "Individualistic customer seeking personal electronics",
        "customer_text": "I want the latest smartphone with best camera for my travels",
        "region": "North America",
        "expected_response": {
            "feature_focus": ["camera_quality", "portability"],
            "social_proof_used": False,
            "personal_benefits_emphasized": True
        },
        "cultural_profile": {
            "individualism": 0.9,
            "power_distance": 0.3,
            "uncertainty_avoidance": 0.4,
            "masculinity": 0.6,
            "long_term_orientation": 0.5,
            "indulgence": 0.7
        }
    },
    
    # Upselling Scenarios
    {
        "scenario": "Family-oriented purchase in collectivist culture",
        "customer_text": "We need a new car for our family of five",
        "region": "Latin America",
        "expected_response": {
            "family_benefits_highlighted": True,
            "group_discount_offered": True,
            "safety_features_emphasized": True
        },
        "cultural_profile": {
            "individualism": 0.3,
            "power_distance": 0.7,
            "uncertainty_avoidance": 0.8,
            "masculinity": 0.6,
            "long_term_orientation": 0.6,
            "indulgence": 0.7
        }
    },
    
    # Return/Exchange Scenarios
    {
        "scenario": "Return request in high uncertainty avoidance culture",
        "customer_text": "This product doesn't match the description. I want to return it.",
        "region": "Germany",
        "expected_response": {
            "process_clarity": 0.9,
            "policy_reference": True,
            "assurance_provided": True
        },
        "cultural_profile": {
            "individualism": 0.7,
            "power_distance": 0.4,
            "uncertainty_avoidance": 0.8,
            "masculinity": 0.7,
            "long_term_orientation": 0.8,
            "indulgence": 0.4
        }
    }
]

# Expected improvements from Delight.AI SEDS
EXPECTED_IMPROVEMENTS = {
    "customer_satisfaction": {
        "baseline": 0.65,
        "target": 0.85,
        "metric": "CSAT Score (1-5 scale)"
    },
    "response_appropriateness": {
        "baseline": 0.60,
        "target": 0.88,
        "metric": "% of culturally appropriate responses"
    },
    "issue_resolution_rate": {
        "baseline": 0.70,
        "target": 0.92,
        "metric": "First Contact Resolution Rate"
    },
    "upsell_success": {
        "baseline": 0.15,
        "target": 0.35,
        "metric": "% of successful upsells"
    },
    "return_rate": {
        "baseline": 0.12,
        "target": 0.08,
        "metric": "Product return rate"
    }
}
