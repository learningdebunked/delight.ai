"""
Stress testing for the cultural adaptation system.

This module contains stress tests to evaluate the system's behavior under extreme conditions,
including high load, edge cases, and failure scenarios.
"""

import unittest
import numpy as np
import time
import random
import os
import gc
import psutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.enhanced_cultural_model import (
    CulturalAdaptationEngine,
    CulturalProfile,
    CulturalDimension,
    ExpertValidationSystem
)
from models.validation.validator import StatisticalTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StressTestResult:
    """Container for stress test results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.metrics: Dict[str, Any] = {}
        self.failures: List[Dict[str, Any]] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.memory_usage: List[float] = []
        
    def start_timer(self):
        """Start the test timer."""
        self.start_time = time.time()
        
    def stop_timer(self):
        """Stop the test timer."""
        self.end_time = time.time()
        self.metrics['duration_seconds'] = self.end_time - self.start_time
    
    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / (1024 * 1024))  # MB
    
    def add_metric(self, name: str, value: Any):
        """Add a metric to the test results."""
        self.metrics[name] = value
    
    def add_failure(self, message: str, details: Any = None):
        """Record a test failure."""
        self.failures.append({
            'message': message,
            'details': str(details) if details is not None else None
        })
    
    def finalize(self):
        """Finalize the test results."""
        self.passed = len(self.failures) == 0
        self.stop_timer()
        
        # Calculate memory statistics
        if self.memory_usage:
            self.metrics['memory_usage_mb'] = {
                'min': min(self.memory_usage),
                'max': max(self.memory_usage),
                'avg': sum(self.memory_usage) / len(self.memory_usage)
            }
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'metrics': self.metrics,
            'failures': self.failures,
            'duration_seconds': self.metrics.get('duration_seconds', 0)
        }


class CulturalStressTester:
    """Stress testing framework for the cultural adaptation system."""
    
    def __init__(self):
        """Initialize the stress tester."""
        self.engine = CulturalAdaptationEngine()
        self.results: List[StressTestResult] = []
        
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all stress tests and return results."""
        test_methods = [
            self.test_high_dimensional_profiles,
            self.test_high_frequency_requests,
            self.test_memory_usage,
            self.test_concurrent_access,
            self.test_invalid_inputs,
            self.test_resource_exhaustion,
            self.test_recovery_from_failure
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"Running stress test: {test_method.__name__}")
                result = test_method()
                self.results.append(result)
                
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"Test {test_method.__name__} {status} in {result.metrics.get('duration_seconds', 0):.2f}s")
                
            except Exception as e:
                logger.error(f"Error running test {test_method.__name__}: {str(e)}")
                result = StressTestResult(test_method.__name__)
                result.add_failure(f"Test raised exception: {str(e)}")
                result.finalize()
                self.results.append(result)
        
        return [r.to_dict() for r in self.results]
    
    def test_high_dimensional_profiles(self) -> StressTestResult:
        """Test with a large number of cultural dimensions."""
        result = StressTestResult('test_high_dimensional_profiles')
        result.start_timer()
        
        try:
            # Create a profile with many dimensions
            num_dimensions = 1000
            dimensions = {f'dim_{i}': random.random() for i in range(num_dimensions)}
            
            # Create and add profile
            profile = CulturalProfile(
                profile_id='high_dim',
                name='High Dimensional Profile',
                dimensions=dimensions
            )
            
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Time profile creation and addition
            add_start = time.time()
            self.engine.add_profile(profile)
            add_time = time.time() - add_start
            
            # Check memory usage
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            mem_used = end_mem - start_mem
            
            # Record metrics
            result.add_metric('num_dimensions', num_dimensions)
            result.add_metric('add_time_seconds', add_time)
            result.add_metric('memory_used_mb', mem_used)
            
            # Verify profile was added
            retrieved = self.engine.get_profile('high_dim')
            if retrieved is None:
                result.add_failure("Failed to retrieve high-dimensional profile")
            
            result.record_memory_usage()
            
        except Exception as e:
            result.add_failure(f"High-dimensional test failed: {str(e)}")
        
        return result.finalize()
    
    def test_high_frequency_requests(self) -> StressTestResult:
        """Test handling of high-frequency adaptation requests."""
        result = StressTestResult('test_high_frequency_requests')
        result.start_timer()
        
        try:
            # Create test profiles
            num_profiles = 100
            profiles = []
            
            for i in range(num_profiles):
                profile = CulturalProfile(
                    profile_id=f'test_{i}',
                    name=f'Test Profile {i}',
                    dimensions={dim: random.random() for dim in CulturalDimension}
                )
                self.engine.add_profile(profile)
                profiles.append(profile)
            
            # Simulate high-frequency requests
            num_requests = 1000
            request_times = []
            
            for _ in range(num_requests):
                src, tgt = random.sample(profiles, 2)
                
                start_time = time.time()
                plan = self.engine.get_adaptation_plan(
                    source_id=src.profile_id,
                    target_id=tgt.profile_id,
                    context={'stress_test': True}
                )
                request_time = time.time() - start_time
                request_times.append(request_time)
                
                # Verify plan is valid
                if not plan or not hasattr(plan, 'dimensions'):
                    result.add_failure("Invalid adaptation plan returned")
            
            # Calculate statistics
            result.add_metric('num_requests', num_requests)
            result.add_metric('avg_request_time_ms', np.mean(request_times) * 1000)
            result.add_metric('p95_request_time_ms', np.percentile(request_times, 95) * 1000)
            result.add_metric('max_request_time_ms', max(request_times) * 1000)
            
            # Check for timeouts (arbitrary 100ms threshold)
            slow_requests = [t for t in request_times if t > 0.1]
            if slow_requests:
                result.add_failure(
                    f"{len(slow_requests)} requests exceeded 100ms threshold",
                    f"Max time: {max(slow_requests)*1000:.2f}ms"
                )
            
            result.record_memory_usage()
            
        except Exception as e:
            result.add_failure(f"High-frequency request test failed: {str(e)}")
        
        return result.finalize()
    
    def test_memory_usage(self) -> StressTestResult:
        """Test memory usage with many profiles and adaptations."""
        result = StressTestResult('test_memory_usage')
        result.start_timer()
        
        try:
            # Record initial memory
            initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Create many profiles
            num_profiles = 1000
            profile_ids = []
            
            for i in range(num_profiles):
                profile = CulturalProfile(
                    profile_id=f'mem_test_{i}',
                    name=f'Memory Test {i}',
                    dimensions={dim: random.random() for dim in CulturalDimension}
                )
                self.engine.add_profile(profile)
                profile_ids.append(profile.profile_id)
                
                # Record memory every 100 profiles
                if i % 100 == 0:
                    result.record_memory_usage()
            
            # Force garbage collection
            gc.collect()
            
            # Record final memory
            final_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            mem_used = final_mem - initial_mem
            
            # Calculate memory per profile
            mem_per_profile = mem_used / num_profiles
            
            # Record metrics
            result.add_metric('num_profiles', num_profiles)
            result.add_metric('total_memory_used_mb', mem_used)
            result.add_metric('memory_per_profile_kb', mem_per_profile * 1024)
            
            # Check for excessive memory usage (> 1KB per profile)
            if mem_per_profile > 1.0 / 1024:  # 1KB per profile
                result.add_failure(
                    f"High memory usage per profile: {mem_per_profile*1024:.2f} KB",
                    "Expected < 1 KB per profile"
                )
            
        except Exception as e:
            result.add_failure(f"Memory usage test failed: {str(e)}")
        
        return result.finalize()
    
    def test_concurrent_access(self) -> StressTestResult:
        """Test thread safety with concurrent access."""
        result = StressTestResult('test_concurrent_access')
        result.start_timer()
        
        try:
            # Create test profiles
            num_profiles = 100
            profiles = []
            
            for i in range(num_profiles):
                profile = CulturalProfile(
                    profile_id=f'concurrent_{i}',
                    name=f'Concurrent Test {i}',
                    dimensions={dim: random.random() for dim in CulturalDimension}
                )
                self.engine.add_profile(profile)
                profiles.append(profile)
            
            # Number of concurrent workers
            num_workers = 10
            num_requests_per_worker = 50
            
            def worker(worker_id):
                """Worker function for concurrent testing."""
                results = []
                for _ in range(num_requests_per_worker):
                    src, tgt = random.sample(profiles, 2)
                    
                    try:
                        start_time = time.time()
                        plan = self.engine.get_adaptation_plan(
                            source_id=src.profile_id,
                            target_id=tgt.profile_id,
                            context={'worker': worker_id}
                        )
                        request_time = time.time() - start_time
                        
                        results.append({
                            'success': True,
                            'time': request_time,
                            'plan_valid': plan is not None
                        })
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e)
                        })
                return results
            
            # Run workers in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker, i) 
                    for i in range(num_workers)
                ]
                
                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())
            
            # Analyze results
            total_requests = len(all_results)
            successful = sum(1 for r in all_results if r['success'])
            failed = total_requests - successful
            
            # Get timing for successful requests
            request_times = [
                r['time'] for r in all_results 
                if r['success'] and 'time' in r
            ]
            
            # Record metrics
            result.add_metric('total_requests', total_requests)
            result.add_metric('successful_requests', successful)
            result.add_metric('failed_requests', failed)
            
            if request_times:
                result.add_metric('avg_request_time_ms', np.mean(request_times) * 1000)
                result.add_metric('max_request_time_ms', max(request_times) * 1000)
            
            # Check for failures
            if failed > 0:
                # Get error messages from failed requests
                errors = {}
                for r in all_results:
                    if not r['success']:
                        err = r.get('error', 'unknown')
                        errors[err] = errors.get(err, 0) + 1
                
                result.add_failure(
                    f"{failed} out of {total_requests} requests failed",
                    dict(errors)
                )
            
            result.record_memory_usage()
            
        except Exception as e:
            result.add_failure(f"Concurrent access test failed: {str(e)}")
        
        return result.finalize()
    
    def test_invalid_inputs(self) -> StressTestResult:
        """Test handling of invalid inputs and edge cases."""
        result = StressTestResult('test_invalid_inputs')
        result.start_timer()
        
        test_cases = [
            # (description, test_function, expected_exception)
            (
                "None profile ID",
                lambda: self.engine.get_profile(None),
                (ValueError, TypeError)
            ),
            (
                "Empty profile ID",
                lambda: self.engine.get_profile(''),
                ValueError
            ),
            (
                "Very long profile ID",
                lambda: self.engine.get_profile('x' * 10000),
                (ValueError, KeyError)
            ),
            (
                "Non-existent profile",
                lambda: self.engine.get_adaptation_plan(
                    source_id='nonexistent1',
                    target_id='nonexistent2',
                    context={}
                ),
                (ValueError, KeyError)
            ),
            (
                "Invalid dimension values",
                lambda: CulturalProfile(
                    profile_id='invalid_dims',
                    name='Invalid Dimensions',
                    dimensions={dim: 100.0 for dim in CulturalDimension}  # Values > 1.0
                ),
                (ValueError, AssertionError)
            )
        ]
        
        for desc, test_func, expected_exc in test_cases:
            try:
                test_func()
                result.add_failure(
                    f"Expected exception not raised",
                    f"Test case: {desc}"
                )
            except expected_exc:
                # Expected exception
                pass
            except Exception as e:
                result.add_failure(
                    f"Unexpected exception: {type(e).__name__}",
                    f"Test case: {desc}\nError: {str(e)}"
                )
        
        return result.finalize()
    
    def test_resource_exhaustion(self) -> StressTestResult:
        """Test behavior when system resources are exhausted."""
        result = StressTestResult('test_resource_exhaustion')
        result.start_timer()
        
        try:
            # Test with extremely large profile
            large_dimensions = {f'dim_{i}': 0.5 for i in range(1000000)}
            
            try:
                profile = CulturalProfile(
                    profile_id='huge_profile',
                    name='Huge Profile',
                    dimensions=large_dimensions
                )
                self.engine.add_profile(profile)
                
                # If we get here, the test failed to detect resource exhaustion
                result.add_failure(
                    "System accepted extremely large profile",
                    "Expected resource limit exception"
                )
                
            except (MemoryError, ValueError, RuntimeError):
                # Expected behavior
                pass
            
            # Test with too many profiles
            max_profiles = 10000  # Arbitrary limit
            for i in range(max_profiles + 100):
                try:
                    profile = CulturalProfile(
                        profile_id=f'limit_test_{i}',
                        name=f'Limit Test {i}',
                        dimensions={dim: 0.5 for dim in CulturalDimension}
                    )
                    self.engine.add_profile(profile)
                except Exception as e:
                    if i < max_profiles // 2:
                        # Failed too early
                        result.add_failure(
                            f"Failed to add profile {i} (limit: {max_profiles})",
                            str(e)
                        )
                    break
            
            result.record_memory_usage()
            
        except Exception as e:
            result.add_failure(f"Resource exhaustion test failed: {str(e)}")
        
        return result.finalize()
    
    def test_recovery_from_failure(self) -> StressTestResult:
        """Test system recovery after failures."""
        result = StressTestResult('test_recovery_from_failure')
        result.start_timer()
        
        try:
            # Create a test profile
            profile = CulturalProfile(
                profile_id='recovery_test',
                name='Recovery Test',
                dimensions={dim: 0.5 for dim in CulturalDimension}
            )
            self.engine.add_profile(profile)
            
            # Simulate a failure (e.g., by corrupting internal state)
            if hasattr(self.engine, '_profiles'):
                # This is a bit hacky, but it simulates corruption
                original_profiles = self.engine._profiles.copy()
                self.engine._profiles = None
                
                try:
                    # This should fail
                    self.engine.get_profile('recovery_test')
                    result.add_failure("System did not fail on corrupted state")
                except Exception:
                    # Expected failure
                    pass
                
                # Restore state
                self.engine._profiles = original_profiles
                
                # Verify recovery
                try:
                    recovered = self.engine.get_profile('recovery_test')
                    if recovered is None:
                        result.add_failure("Failed to recover profile after corruption")
                except Exception as e:
                    result.add_failure("Failed to recover from corrupted state", str(e))
            
            result.record_memory_usage()
            
        except Exception as e:
            result.add_failure(f"Recovery test failed: {str(e)}")
        
        return result.finalize()


def run_stress_tests():
    """Run all stress tests and generate a report."""
    import json
    from datetime import datetime
    import os
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"reports/stress_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tester
    tester = CulturalStressTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Save results to file
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        },
        'results': results
    }
    
    # Save full report
    report_path = os.path.join(output_dir, 'stress_test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    summary = {
        'total_tests': len(results),
        'passed': sum(1 for r in results if r.get('passed', False)),
        'failed': sum(1 for r in results if not r.get('passed', True)),
        'test_duration_seconds': sum(r.get('duration_seconds', 0) for r in results),
        'failures': [
            {
                'test': r['test_name'],
                'errors': [f['message'] for f in r.get('failures', [])]
            }
            for r in results if not r.get('passed', True)
        ]
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Stress Test Summary\n\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Tests**: {summary['total_tests']}\n")
        f.write(f"- **Passed**: {summary['passed']}\n")
        f.write(f"- **Failed**: {summary['failed']}\n")
        f.write(f"- **Total Duration**: {summary['test_duration_seconds']:.2f} seconds\n\n")
        
        if summary['failures']:
            f.write("## Failures\n\n")
            for failure in summary['failures']:
                f.write(f"### {failure['test']}\n")
                for error in failure['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")
    
    print(f"Stress tests completed. Report saved to {report_path}")
    print(f"Summary: {summary['passed']} passed, {summary['failed']} failed")
    
    return summary


if __name__ == "__main__":
    import sys
    run_stress_tests()
