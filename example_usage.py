"""
Example usage of the enhanced SEDS system with multi-modal processing and performance tracking.
"""
import torch
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment and create necessary directories."""
    # Create directories if they don't exist
    Path("data/audio").mkdir(parents=True, exist_ok=True)
    Path("data/images").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

def example_multimodal_processing():
    """Demonstrate multi-modal processing."""
    from models.multimodal_processor import MultiModalProcessor
    
    logger.info("Initializing multi-modal processor...")
    processor = MultiModalProcessor(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Example text input
    text = "Hello, how can I help you today?"
    
    # Example usage with just text
    logger.info("Processing text input...")
    features, fused = processor(text=text)
    logger.info(f"Text features shape: {features.text.shape if features.text is not None else 'None'}")
    logger.info(f"Fused features shape: {fused.shape}")
    
    # Example with all modalities (assuming files exist)
    try:
        audio_path = "data/audio/example.wav"
        image_path = "data/images/example.jpg"
        
        logger.info("Processing multi-modal input...")
        features, fused = processor(
            text=text,
            audio_path=audio_path,
            image_path=image_path
        )
        logger.info(f"All features processed. Fused shape: {fused.shape}")
    except FileNotFoundError as e:
        logger.warning(f"Could not process all modalities: {e}")

def example_performance_tracking():
    """Demonstrate performance tracking."""
    from models.performance_tracker import PerformanceTracker
    import time
    import random
    
    logger.info("Initializing performance tracker...")
    tracker = PerformanceTracker(history_file="results/performance_history.jsonl")
    
    # Simulate recording metrics
    for i in range(10):
        metrics = {
            'accuracy': random.uniform(0.8, 0.99),
            'latency_ms': random.uniform(50, 200),
            'memory_mb': random.uniform(100, 500)
        }
        
        tracker.record_metrics(
            metrics=metrics,
            metadata={"iteration": i, "phase": "training"}
        )
        time.sleep(0.1)  # Simulate processing time
    
    # Get and log summary
    summary = tracker.get_summary()
    logger.info("Performance summary:")
    for metric, stats in summary.items():
        if metric != 'system':
            logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

def example_theorems_validation():
    """Demonstrate theorem validation."""
    from models.theorems import SEDSTheorems, ConvergenceMetrics
    import numpy as np
    
    logger.info("Validating convergence theorem...")
    
    # Simulate training loss history (should be decreasing)
    loss_history = [1.0 / (i + 1) for i in range(100)]
    
    # Create metrics
    metrics = ConvergenceMetrics(
        loss_history=loss_history,
        gradient_norms=[loss_history[i] - loss_history[i+1] for i in range(len(loss_history)-1)]
    )
    
    # Validate convergence
    result = SEDSTheorems.validate_convergence(metrics)
    logger.info(f"Converged: {result.is_converged}")
    logger.info(f"Convergence rate: {result.convergence_rate:.4f}")

def example_benchmark():
    """Demonstrate the ServiceExcellence benchmark."""
    from benchmark import ServiceExcellenceBench, TestCase, evaluate_service_excellence
    
    logger.info("Running ServiceExcellence benchmark...")
    
    # Create a simple mock model for demonstration
    class MockModel:
        def __call__(self, text, user_context):
            import time
            start_time = time.time()
            
            # Simulate processing
            time.sleep(0.1)
            
            return {
                'label': 'greeting_response',
                'emotion': 'positive',
                'response_time': time.time() - start_time,
                'cultural_score': 0.9,
                'similarity_score': 0.85
            }
            
        def eval(self):
            """Set the model to evaluation mode."""
            return self
    
    # Create benchmark with test cases
    test_cases = [
        TestCase(
            input_data={
                'text': 'Hello, how are you?',
                'user_context': {'language': 'en'}
            },
            expected_output={
                'label': 'greeting_response',
                'emotion': 'positive'
            },
            metadata={'id': 'test_1'}
        )
    ]
    
    benchmark = ServiceExcellenceBench(test_cases=test_cases)
    
    # Run evaluation
    results = evaluate_service_excellence(
        model=MockModel(),
        benchmark=benchmark,
        output_dir="results/benchmark"
    )
    
    logger.info("Benchmark results:")
    if 'aggregate' in results:
        for metric, stats in results['aggregate'].items():
            if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
                logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            else:
                logger.info(f"  {metric}: {stats}")
    else:
        logger.info("  No aggregate results available")
        logger.info(f"  Raw results: {results}")

def main():
    """Run all example use cases."""
    setup_environment()
    
    logger.info("=== Starting SEDS System Demo ===")
    
    # Run examples
    example_multimodal_processing()
    example_performance_tracking()
    example_theorems_validation()
    example_benchmark()
    
    logger.info("=== Demo Completed ===")

if __name__ == "__main__":
    main()
