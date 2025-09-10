"""
Test cases for edge cases and theoretical bounds in the cultural adaptation system.
"""

import unittest
import numpy as np
from typing import Dict, Any
from models.enhanced_cultural_model import CulturalAdaptationEngine, CulturalProfile, CulturalDimension
from models.cultural_adaptation_utils import CulturalAdapter

class TestCulturalEdgeCases(unittest.TestCase):
    """Test cases for edge cases and theoretical bounds in cultural adaptation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = CulturalAdaptationEngine()
        self.adapter = CulturalAdapter()
        
        # Create test profiles
        self.neutral_profile = CulturalProfile(
            profile_id='neutral',
            name='Neutral Profile',
            dimensions={dim: 0.5 for dim in CulturalDimension}
        )
        
        self.extreme_profile = CulturalProfile(
            profile_id='extreme',
            name='Extreme Profile',
            dimensions={dim: 1.0 for dim in CulturalDimension}  # All dimensions at max
        )
        
        self.engine.add_profile(self.neutral_profile)
        self.engine.add_profile(self.extreme_profile)
    
    def test_extreme_cultural_distance(self):
        """Test adaptation with maximum cultural distance."""
        # Get adaptation plan between neutral and extreme profiles
        plan = self.engine.get_adaptation_plan(
            source_id='neutral',
            target_id='extreme',
            context={'domain': 'test', 'urgency': 1.0}
        )
        
        # Verify all dimensions have maximum adaptation
        for dim in CulturalDimension:
            self.assertIn(dim, plan.dimensions)
            self.assertAlmostEqual(plan.dimensions[dim].target, 1.0, delta=0.01)
    
    def test_identical_profiles(self):
        """Test adaptation with identical source and target profiles."""
        plan = self.engine.get_adaptation_plan(
            source_id='neutral',
            target_id='neutral',  # Same as source
            context={'domain': 'test'}
        )
        
        # Verify no adaptation needed
        for dim in CulturalDimension:
            self.assertIn(dim, plan.dimensions)
            self.assertAlmostEqual(plan.dimensions[dim].strength, 0.0, delta=0.01)
    
    def test_missing_profile(self):
        """Test behavior when a profile is missing."""
        with self.assertRaises(ValueError):
            self.engine.get_adaptation_plan(
                source_id='nonexistent',
                target_id='neutral',
                context={}
            )
    
    def test_empty_context(self):
        """Test adaptation with empty context."""
        plan = self.engine.get_adaptation_plan(
            source_id='neutral',
            target_id='extreme',
            context={}  # Empty context
        )
        self.assertIsNotNone(plan)
    
    def test_extreme_text_adaptation(self):
        """Test text adaptation with extreme values."""
        # Test with empty string
        result, _ = self.adapter.adapt_formality("", 1.0)
        self.assertEqual(result, "")
        
        # Test with very long string
        long_text = "hello " * 1000
        result, confidence = self.adapter.adapt_formality(long_text, 1.0)
        self.assertGreater(len(result), 0)
        self.assertGreaterEqual(confidence, 0.0)
        
        # Test with special characters
        special_text = "!@#$%^&*()_+{}|:<>?\""
        result, _ = self.adapter.adapt_formality(special_text, 0.5)
        self.assertEqual(result, special_text)
    
    def test_invalid_confidence_values(self):
        """Test with invalid confidence values."""
        # Values outside [0,1] range
        with self.assertRaises(ValueError):
            self.engine.get_adaptation_plan(
                source_id='neutral',
                target_id='extreme',
                context={'confidence': -0.1}
            )
        
        with self.assertRaises(ValueError):
            self.engine.get_adaptation_plan(
                source_id='neutral',
                target_id='extreme',
                context={'confidence': 1.1}
            )
    
    def test_high_dimensional_adaptation(self):
        """Test with high-dimensional cultural profiles."""
        # Create a profile with many dimensions
        many_dims = {f'dim_{i}': 0.5 for i in range(1000)}
        profile = CulturalProfile(
            profile_id='high_dim',
            name='High Dimensional Profile',
            dimensions=many_dims
        )
        self.engine.add_profile(profile)
        
        plan = self.engine.get_adaptation_plan(
            source_id='neutral',
            target_id='high_dim',
            context={}
        )
        self.assertEqual(len(plan.dimensions), len(many_dims))
    
    def test_concurrent_adaptation(self):
        """Test concurrent adaptation requests."""
        import threading
        
        results = []
        
        def adapt():
            try:
                plan = self.engine.get_adaptation_plan(
                    source_id='neutral',
                    target_id='extreme',
                    context={'thread': threading.get_ident()}
                )
                results.append(True)
            except Exception:
                results.append(False)
        
        # Start multiple threads
        threads = [threading.Thread(target=adapt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All adaptations should succeed
        self.assertTrue(all(results))
        self.assertEqual(len(results), 10)


if __name__ == '__main__':
    unittest.main()
