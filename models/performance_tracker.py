"""
Performance tracking and monitoring for the SEDS system.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import json
import pandas as pd

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'metrics': self.metrics,
            'metadata': self.metadata
        }

class PerformanceTracker:
    """Tracks and analyzes performance metrics for the SEDS system."""
    
    def __init__(self, window_size: int = 100, history_file: Optional[str] = None):
        """
        Initialize the performance tracker.
        
        Args:
            window_size: Size of the sliding window for moving averages
            history_file: Optional file to persist history
        """
        self.window_size = window_size
        self.history_file = history_file
        self.metrics_history: List[PerformanceMetrics] = []
        self.moving_averages: Dict[str, float] = {}
        self.last_update_time = time.time()
        self.start_time = time.time()
        
        # Track resource usage
        self.resource_usage = {
            'cpu': deque(maxlen=window_size),
            'memory': deque(maxlen=window_size),
            'latency': deque(maxlen=window_size)
        }
        
        # Load history if file exists
        if history_file:
            self.load_history()
    
    def record_metrics(self, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a new set of metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            metadata: Optional additional context
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp
        current_time = time.time()
        metadata['elapsed_time'] = current_time - self.start_time
        
        # Create and store metrics
        metric_entry = PerformanceMetrics(
            timestamp=current_time,
            metrics=metrics,
            metadata=metadata
        )
        self.metrics_history.append(metric_entry)
        
        # Update moving averages
        self._update_moving_averages(metrics)
        
        # Save to history file if configured
        if self.history_file:
            self._save_to_history(metric_entry)
    
    def _update_moving_averages(self, metrics: Dict[str, float]) -> None:
        """Update moving averages for all metrics."""
        for key, value in metrics.items():
            if key not in self.moving_averages:
                self.moving_averages[key] = value
            else:
                # Exponential moving average
                alpha = 2.0 / (self.window_size + 1)
                self.moving_averages[key] = (1 - alpha) * self.moving_averages[key] + alpha * value
    
    def get_moving_average(self, metric_name: str) -> Optional[float]:
        """Get the current moving average for a metric."""
        return self.moving_averages.get(metric_name)
    
    def get_metric_history(self, metric_name: str) -> List[Tuple[float, float]]:
        """Get the history of a specific metric as (timestamp, value) pairs."""
        return [
            (m.timestamp, m.metrics[metric_name])
            for m in self.metrics_history
            if metric_name in m.metrics
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of all recorded metrics."""
        if not self.metrics_history:
            return {}
            
        # Get all metric names
        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update(entry.metrics.keys())
        
        # Calculate statistics for each metric
        summary = {}
        for metric in all_metrics:
            values = [m.metrics[metric] for m in self.metrics_history if metric in m.metrics]
            if values:
                summary[metric] = {
                    'count': len(values),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'latest': values[-1],
                    'moving_average': self.moving_averages.get(metric)
                }
        
        # Add system metrics
        summary['system'] = {
            'total_metrics': len(self.metrics_history),
            'tracking_duration': time.time() - self.start_time,
            'metrics_per_second': len(self.metrics_history) / max(1, (time.time() - self.start_time))
        }
        
        return summary
    
    def detect_anomalies(self, metric_name: str, threshold_std: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a metric's history.
        
        Args:
            metric_name: Name of the metric to analyze
            threshold_std: Number of standard deviations to consider as anomaly
            
        Returns:
            List of detected anomalies with timestamps and values
        """
        values = [m.metrics[metric_name] for m in self.metrics_history if metric_name in m.metrics]
        if not values:
            return []
            
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        for entry in self.metrics_history:
            if metric_name in entry.metrics:
                value = entry.metrics[metric_name]
                if abs(value - mean) > threshold_std * std:
                    anomalies.append({
                        'timestamp': entry.timestamp,
                        'value': value,
                        'deviation': (value - mean) / std if std != 0 else 0,
                        'metadata': entry.metadata
                    })
        
        return anomalies
    
    def _save_to_history(self, metrics: PerformanceMetrics) -> None:
        """Save metrics to history file."""
        if not self.history_file:
            return
            
        try:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            print(f"Warning: Failed to save metrics to history: {e}")
    
    def load_history(self) -> None:
        """Load metrics history from file."""
        if not self.history_file:
            return
            
        try:
            with open(self.history_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        metrics = PerformanceMetrics(
                            timestamp=data['timestamp'],
                            metrics=data['metrics'],
                            metadata=data.get('metadata', {})
                        )
                        self.metrics_history.append(metrics)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            # File doesn't exist yet, that's fine
            pass
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to a pandas DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
            
        # Create a list of all metrics
        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update(entry.metrics.keys())
        
        # Create rows for DataFrame
        rows = []
        for entry in self.metrics_history:
            row = {
                'timestamp': entry.timestamp,
                'datetime': datetime.fromtimestamp(entry.timestamp),
                **{f'meta_{k}': v for k, v in entry.metadata.items()}
            }
            row.update(entry.metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics_history = []
        self.moving_averages = {}
        self.last_update_time = time.time()
        self.start_time = time.time()
        for key in self.resource_usage:
            self.resource_usage[key].clear()
