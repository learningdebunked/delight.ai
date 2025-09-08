"""
ServiceExcellence-Bench: A comprehensive benchmarking framework for evaluating
service excellence in AI systems.
"""
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """A single test case for evaluation."""
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceExcellenceBench(Dataset):
    """
    A benchmark for evaluating service excellence in AI systems.
    """
    
    def __init__(
        self,
        test_cases: Optional[List[TestCase]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the benchmark.
        
        Args:
            test_cases: List of test cases
            data_dir: Directory containing test data (alternative to test_cases)
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
        """
        self.test_cases = test_cases or []
        self.batch_size = batch_size
        self.device = device
        
        if data_dir is not None:
            self._load_from_directory(data_dir)
    
    def _load_from_directory(self, data_dir: Union[str, Path]) -> None:
        """Load test cases from a directory."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
            
        # Load test cases from JSON files
        for json_file in data_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for item in data:
                        self.test_cases.append(TestCase(**item))
                else:
                    self.test_cases.append(TestCase(**data))
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a single test case to the benchmark."""
        self.test_cases.append(test_case)
    
    def add_test_cases(self, test_cases: List[TestCase]) -> None:
        """Add multiple test cases to the benchmark."""
        self.test_cases.extend(test_cases)
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __getitem__(self, idx: int) -> TestCase:
        return self.test_cases[idx]
    
    def evaluate(
        self,
        model: Any,
        metrics: List[callable],
        output_dir: Optional[Union[str, Path]] = None,
        save_results: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model on the benchmark.
        
        Args:
            model: The model to evaluate
            metrics: List of metric functions to compute
            output_dir: Directory to save results
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary of metric results
        """
        if not self.test_cases:
            raise ValueError("No test cases available for evaluation")
            
        results = {
            'per_case': [],
            'aggregate': {}
        }
        
        # Create data loader
        dataloader = DataLoader(
            self.test_cases,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )
        
        # Run evaluation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch_results = self._evaluate_batch(batch, model, metrics)
                results['per_case'].extend(batch_results)
        
        # Compute aggregate metrics
        results['aggregate'] = self._aggregate_metrics(results['per_case'], metrics)
        
        # Save results if requested
        if save_results and output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def _evaluate_batch(
        self,
        batch: List[TestCase],
        model: Any,
        metrics: List[callable]
    ) -> List[Dict[str, Any]]:
        """Evaluate a single batch of test cases."""
        batch_results = []
        
        for test_case in batch:
            try:
                # Run model inference
                output = model(**test_case.input_data)
                
                # Compute metrics
                case_result = {
                    'test_case': test_case.metadata.get('id', str(id(test_case))),
                    'metrics': {}
                }
                
                for metric in metrics:
                    try:
                        metric_name = metric.__name__
                        metric_value = metric(output, test_case.expected_output)
                        case_result['metrics'][metric_name] = float(metric_value)
                    except Exception as e:
                        logger.warning(f"Error computing metric {metric.__name__}: {e}")
                        case_result['metrics'][metric_name] = float('nan')
                
                batch_results.append(case_result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case: {e}", exc_info=True)
                batch_results.append({
                    'test_case': test_case.metadata.get('id', str(id(test_case))),
                    'error': str(e),
                    'metrics': {metric.__name__: float('nan') for metric in metrics}
                })
        
        return batch_results
    
    def _aggregate_metrics(
        self,
        results: List[Dict[str, Any]],
        metrics: List[callable]
    ) -> Dict[str, Dict[str, float]]:
        """Compute aggregate metrics across all test cases."""
        if not results:
            return {}
            
        # Initialize aggregates
        metric_names = [metric.__name__ for metric in metrics]
        aggregates = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }
        
        # Compute statistics for each metric
        for metric_name in metric_names:
            values = []
            for result in results:
                if 'metrics' in result and metric_name in result['metrics']:
                    value = result['metrics'][metric_name]
                    if not np.isnan(value):
                        values.append(value)
            
            if values:
                aggregates['mean'][metric_name] = float(np.mean(values))
                aggregates['std'][metric_name] = float(np.std(values))
                aggregates['min'][metric_name] = float(np.min(values))
                aggregates['max'][metric_name] = float(np.max(values))
            else:
                aggregates['mean'][metric_name] = float('nan')
                aggregates['std'][metric_name] = float('nan')
                aggregates['min'][metric_name] = float('nan')
                aggregates['max'][metric_name] = float('nan')
        
        return aggregates
    
    def _save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        """Save evaluation results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for the run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save per-case results
        per_case_file = output_dir / f"results_per_case_{timestamp}.json"
        with open(per_case_file, 'w') as f:
            json.dump(results['per_case'], f, indent=2)
        
        # Save aggregate results
        agg_file = output_dir / f"results_aggregate_{timestamp}.json"
        with open(agg_file, 'w') as f:
            json.dump(results['aggregate'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


# Common metrics for service excellence
def accuracy_score(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Compute accuracy between predicted and expected outputs."""
    if 'label' not in predicted or 'label' not in expected:
        return float('nan')
    return float(predicted['label'] == expected['label'])

def response_time(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Get response time from model output."""
    return predicted.get('response_time', float('nan'))

def cultural_appropriateness(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Score for cultural appropriateness (higher is better)."""
    return predicted.get('cultural_score', 0.0)

def emotion_similarity(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Compute similarity between predicted and expected emotions."""
    if 'emotion' not in predicted or 'emotion' not in expected:
        return float('nan')
    
    # Simple 0-1 score for now
    return 1.0 if predicted['emotion'] == expected['emotion'] else 0.0

def semantic_similarity(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Compute semantic similarity between responses."""
    # This would typically use a pre-trained sentence transformer
    # For now, return a placeholder
    return predicted.get('similarity_score', 0.0)


def create_default_benchmark() -> ServiceExcellenceBench:
    """Create a benchmark with default test cases."""
    test_cases = [
        TestCase(
            input_data={
                'text': 'Hello, how are you today?',
                'user_context': {'language': 'en', 'region': 'US'}
            },
            expected_output={
                'label': 'greeting_response',
                'emotion': 'positive',
                'response_time': 0.5,
                'cultural_score': 0.9
            },
            metadata={
                'id': 'greeting_1',
                'description': 'Basic greeting in English',
                'difficulty': 'easy'
            }
        ),
        # Add more test cases as needed
    ]
    
    return ServiceExcellenceBench(test_cases=test_cases)


def evaluate_service_excellence(
    model: Any,
    metrics: Optional[List[callable]] = None,
    benchmark: Optional[ServiceExcellenceBench] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate service excellence.
    
    Args:
        model: The model to evaluate
        metrics: List of metric functions (defaults to common metrics)
        benchmark: Optional benchmark instance (defaults to creating one)
        output_dir: Directory to save results
        
    Returns:
        Evaluation results
    """
    if metrics is None:
        metrics = [
            accuracy_score,
            response_time,
            cultural_appropriateness,
            emotion_similarity,
            semantic_similarity
        ]
    
    if benchmark is None:
        benchmark = create_default_benchmark()
    
    return benchmark.evaluate(
        model=model,
        metrics=metrics,
        output_dir=output_dir
    )
