import unittest
import numpy as np
from numpy.testing import assert_almost_equal

class TestCulturalTheorems(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
    def test_convergence(self):
        """Test Theorem 1: Convergence of cultural adaptation"""
        def adaptation(C):
            return 0.8 * C + 0.1  # Contractive mapping
            
        C = np.random.rand(5)
        prev_norm = np.linalg.norm(C)
        
        for _ in range(100):
            C = 0.5 * C + 0.5 * adaptation(C)
            new_norm = np.linalg.norm(C)
            # Check that the difference decreases
            self.assertLessEqual(abs(new_norm - prev_norm), 1e-6)
            prev_norm = new_norm
    
    def test_invariance(self):
        """Test Theorem 2: Preservation of cultural invariants"""
        def transform(C):
            # Rotation preserves norm
            theta = np.pi/4
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            return R @ C
            
        C = np.array([1, 0])
        original_norm = np.linalg.norm(C)
        transformed = transform(C)
        transformed_norm = np.linalg.norm(transformed)
        
        assert_almost_equal(original_norm, transformed_norm, decimal=6)
    
    def test_fusion_optimality(self):
        """Test Theorem 3: Optimality of cultural fusion"""
        profiles = [np.array([1, 0]), np.array([0, 1])]
        weights = np.array([0.6, 0.4])
        
        # Compute optimal fusion
        optimal = np.average(profiles, axis=0, weights=weights)
        
        # Check it minimizes the weighted sum of squared distances
        def total_distance(C):
            return sum(w * np.sum((C - p)**2) for w, p in zip(weights, profiles))
            
        # Test random points to verify optimality
        for _ in range(10):
            random_point = np.random.rand(2)
            self.assertLessEqual(total_distance(optimal), 
                               total_distance(random_point))

if __name__ == '__main__':
    unittest.main()
