"""
Comprehensive test suite for the cultural adaptation system.
"""

import unittest
import numpy as np
from typing import Dict, List, Any, Optional
import torch
import tempfile
import shutil
import os

from models.enhanced_cultural_model import (
    CulturalAdaptationEngine,
    CulturalProfile,
    CulturalDimension,
    ExpertValidationSystem
)
from models.validation.validator import StatisticalTester
from utils.reproducibility import (
    ReproducibilityConfig,
    configure_reproducibility,
    ExperimentTracker
)

class TestCulturalSystem(unittest.TestCase):
    """Test cases for the cultural adaptation system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all test methods."""
        # Configure reproducibility
        config = ReproducibilityConfig(seed=42)
        configure_reproducibility(config)
        
        # Initialize test profiles
        cls._setup_test_profiles()
        
        # Initialize validation system
        cls.validation_system = ExpertValidationSystem()
        
    @classmethod
    def _setup_test_profiles(cls):
        """Set up test cultural profiles."""
        # Create a neutral profile
        cls.neutral_profile = CulturalProfile(
            profile_id='neutral',
            name='Neutral Profile',
            dimensions={dim: 0.5 for dim in CulturalDimension}
        )
        
        # Create extreme profiles
        cls.extreme_profiles = {}
        for dim in CulturalDimension:
            # Create profile with this dimension at 1.0 and others at 0.0
            dims = {d: 0.0 for d in CulturalDimension}
            dims[dim] = 1.0
            
            profile = CulturalProfile(
                profile_id=f'extreme_{dim.value}',
                name=f'Extreme {dim.value.replace("_", " ").title()}',
                dimensions=dims
            )
            cls.extreme_profiles[dim] = profile
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.engine = CulturalAdaptationEngine()
        
        # Add test profiles to engine
        self.engine.add_profile(self.neutral_profile)
        for profile in self.extreme_profiles.values():
            self.engine.add_profile(profile)
    
    def test_convergence(self):
        """Test that the adaptation process converges."""
        source = self.neutral_profile
        target = self.extreme_profiles[CulturalDimension.INDIVIDUALISM]
        
        # Run adaptation for multiple steps
        prev_distance = float('inf')
        converged = False
        
        for _ in range(100):  # Max 100 iterations
            plan = self.engine.get_adaptation_plan(
                source_id=source.profile_id,
                target_id=target.profile_id,
                context={'domain': 'test_convergence'}
            )
            
            # Apply the adaptation
            new_dims = {}
            for dim, value in source.dimensions.items():
                if dim in plan.dimensions:
                    new_dims[dim] = np.clip(
                        value + plan.dimensions[dim].delta,
                        0.0, 1.0
                    )
                else:
                    new_dims[dim] = value
            
            # Update profile
            source = CulturalProfile(
                profile_id=source.profile_id,
                name=source.name,
                dimensions=new_dims
            )
            
            # Check distance to target
            distance = source.calculate_distance(target)
            
            # Check for convergence
            if abs(distance - prev_distance) < 1e-6:
                converged = True
                break
                
            prev_distance = distance
        
        self.assertTrue(converged, "Adaptation did not converge")
    
    def test_determinism(self):
        """Test that the adaptation is deterministic with the same seed."""
        # Run adaptation twice with the same seed
        results = []
        
        for _ in range(2):
            # Reset seed before each run
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            plan = self.engine.get_adaptation_plan(
                source_id=self.neutral_profile.profile_id,
                target_id=self.extreme_profiles[CulturalDimension.POWER_DISTANCE].profile_id,
                context={'domain': 'test_determinism'}
            )
            
            # Convert plan to dict for comparison
            plan_dict = {
                dim: {
                    'delta': adapt.delta,
                    'strength': adapt.strength,
                    'confidence': adapt.confidence
                }
                for dim, adapt in plan.dimensions.items()
            }
            
            results.append(plan_dict)
        
        # Check that both runs produced identical results
        self.assertEqual(
            results[0], 
            results[1],
            "Adaptation is not deterministic with the same seed"
        )
    
    def test_expert_validation(self):
        """Test expert validation workflow."""
        # Register an expert
        expert_id = "expert_1"
        self.validation_system.register_validator(
            expert_id=expert_id,
            domains=["test_domain"],
            expertise_level=5
        )
        
        # Submit a profile for validation
        profile = CulturalProfile(
            profile_id='test_validation',
            name='Test Validation Profile',
            dimensions={dim: 0.5 for dim in CulturalDimension}
        )
        
        request_id = self.validation_system.submit_for_validation(
            profile=profile,
            priority='high',
            domain='test_domain'
        )
        
        # Assign validation to expert
        task = self.validation_system.assign_validation(expert_id)
        self.assertIsNotNone(task, "No validation task assigned")
        self.assertEqual(task['request_id'], request_id)
        
        # Submit validation feedback
        feedback = {
            'status': 'approved',
            'comments': 'Profile looks good',
            'confidence': 0.9
        }
        
        result = self.validation_system.submit_validation(
            expert_id=expert_id,
            request_id=request_id,
            feedback=feedback
        )
        
        self.assertTrue(result, "Validation submission failed")
        
        # Check that profile was updated
        updated_profile = self.validation_system.get_validated_profile(request_id)
        self.assertEqual(updated_profile.validation_status, 'approved')
    
    def test_statistical_significance(self):
        """Test statistical significance of adaptation improvements."""
        # Generate some test data
        np.random.seed(42)
        
        # Before adaptation scores (lower is better)
        before = np.random.normal(0.5, 0.1, 1000)
        
        # After adaptation scores (simulate improvement)
        after = np.random.normal(0.3, 0.1, 1000)
        
        # Perform t-test
        tester = StatisticalTester()
        result = tester.t_test(before, after)
        
        # Check that the improvement is statistically significant
        self.assertTrue(
            result['significant'],
            f"Improvement not statistically significant (p={result['p_value']:.4f})"
        )
        
        # Check that the effect size is meaningful
        effect = tester.effect_size(before, after)
        self.assertGreater(
            abs(effect['cohens_d']), 
            0.2,
            f"Effect size too small (d={effect['cohens_d']:.3f})"
        )
    
    def test_reproducibility(self):
        """Test experiment reproducibility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two experiment trackers with the same config
            config = {
                'model': 'test_model',
                'learning_rate': 0.001,
                'batch_size': 32,
                'seed': 42
            }
            
            # Run first experiment
            tracker1 = ExperimentTracker(
                output_dir=os.path.join(temp_dir, 'exp1'),
                config=config
            )
            
            with tracker1:
                # Simulate training
                metrics1 = self._simulate_training(tracker1)
            
            # Run second experiment with same config
            tracker2 = ExperimentTracker(
                output_dir=os.path.join(temp_dir, 'exp2'),
                config=config
            )
            
            with tracker2:
                # Simulate training with same random seed
                metrics2 = self._simulate_training(tracker2)
            
            # Check that results are identical
            for (k1, v1), (k2, v2) in zip(metrics1.items(), metrics2.items()):
                self.assertEqual(
                    k1, k2, 
                    f"Metric names don't match: {k1} != {k2}"
                )
                np.testing.assert_almost_equal(
                    v1, v2, 
                    decimal=5,
                    err_msg=f"Metric values don't match for {k1}"
                )
    
    def _simulate_training(self, tracker):
        """Helper method to simulate a training run."""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate some random data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Simulate training loop
        metrics = {}
        for epoch in range(5):
            # Simulate training
            y_pred = np.random.random(100)  # Random predictions
            
            # Calculate metrics
            acc = np.mean((y_pred > 0.5) == y)
            loss = -np.log(y_pred[y == 1]).mean()  # Binary cross-entropy
            
            # Log metrics
            epoch_metrics = {
                'train/accuracy': acc,
                'train/loss': loss,
                'epoch': epoch
            }
            
            if epoch % 2 == 0:
                # Simulate validation
                val_acc = acc * 0.9  # Slightly worse than training
                val_loss = loss * 1.1
                
                epoch_metrics.update({
                    'val/accuracy': val_acc,
                    'val/loss': val_loss
                })
            
            tracker.log_metrics(epoch_metrics, step=epoch)
            
            # Save metrics for comparison
            if epoch == 4:  # Final epoch
                metrics = epoch_metrics
        
        return metrics


if __name__ == '__main__':
    unittest.main()
