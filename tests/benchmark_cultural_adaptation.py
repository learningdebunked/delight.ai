"""
Performance benchmarking for cultural adaptation system.
"""

import time
import random
import statistics
from typing import List, Dict, Tuple
from models.enhanced_cultural_model import CulturalAdaptationEngine, CulturalProfile, CulturalDimension

class CulturalBenchmark:
    """Benchmarking suite for cultural adaptation performance."""
    
    def __init__(self, num_profiles: int = 100):
        """Initialize with test data."""
        self.engine = CulturalAdaptationEngine()
        self.num_profiles = num_profiles
        self._setup_test_profiles()
    
    def _setup_test_profiles(self):
        """Create test profiles with random cultural dimensions."""
        self.profiles = []
        for i in range(self.num_profiles):
            profile = CulturalProfile(
                profile_id=f'test_{i}',
                name=f'Test Profile {i}',
                dimensions={dim: random.random() for dim in CulturalDimension}
            )
            self.engine.add_profile(profile)
            self.profiles.append(profile.profile_id)
    
    def benchmark_adaptation(self, num_requests: int = 1000) -> Dict[str, float]:
        """Benchmark adaptation performance."""
        latencies = []
        
        for _ in range(num_requests):
            # Select random source and target profiles
            source, target = random.sample(self.profiles, 2)
            
            # Time the adaptation
            start = time.perf_counter()
            self.engine.get_adaptation_plan(
                source_id=source,
                target_id=target,
                context={'domain': 'benchmark'}
            )
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        return {
            'total_requests': num_requests,
            'avg_latency_ms': statistics.mean(latencies),
            'p50_latency_ms': statistics.median(latencies),
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[-1],  # 95th percentile
            'p99_latency_ms': statistics.quantiles(latencies, n=100)[-1],  # 99th percentile
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'requests_per_second': num_requests / (sum(latencies) / 1000)
        }
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage with many profiles."""
        import psutil
        import os
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create many profiles
        many_profiles = 1000
        self._setup_test_profiles(many_profiles)
        
        # Measure memory after creating profiles
        after_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        return {
            'baseline_memory_mb': baseline_mem,
            'memory_after_profiles_mb': after_mem,
            'memory_per_profile_kb': ((after_mem - baseline_mem) * 1024) / many_profiles
        }

def run_benchmarks():
    """Run all benchmarks and print results."""
    print("Starting cultural adaptation benchmarks...")
    
    # Initialize benchmark
    benchmark = CulturalBenchmark()
    
    # Run performance benchmark
    print("\n=== Performance Benchmark ===")
    perf_results = benchmark.benchmark_adaptation(1000)
    for metric, value in perf_results.items():
        if 'latency' in metric:
            print(f"{metric}: {value:.2f} ms")
        elif 'per_second' in metric:
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    # Run memory benchmark
    print("\n=== Memory Usage Benchmark ===")
    mem_results = benchmark.benchmark_memory_usage()
    for metric, value in mem_results.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    run_benchmarks()
