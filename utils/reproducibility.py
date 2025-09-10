"""
Reproducibility utilities for deterministic training and evaluation.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any, Union
import json
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class ReproducibilityConfig:
    """Configuration for reproducibility settings."""
    
    def __init__(self, 
                 seed: int = 42,
                 cudnn_deterministic: bool = True,
                 cudnn_benchmark: bool = False,
                 use_deterministic_algorithms: bool = True,
                 warn_only: bool = False):
        """Initialize reproducibility configuration.
        
        Args:
            seed: Random seed for all random number generators
            cudnn_deterministic: If True, sets CuDNN to deterministic mode
            cudnn_benchmark: If True, enables CuDNN benchmark mode (may be faster but less deterministic)
            use_deterministic_algorithms: If True, uses deterministic algorithms where available
            warn_only: If True, only warns about reproducibility issues instead of raising errors
        """
        self.seed = seed
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_benchmark = cudnn_benchmark
        self.use_deterministic_algorithms = use_deterministic_algorithms
        self.warn_only = warn_only
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'seed': self.seed,
            'cudnn_deterministic': self.cudnn_deterministic,
            'cudnn_benchmark': self.cudnn_benchmark,
            'use_deterministic_algorithms': self.use_deterministic_algorithms,
            'warn_only': self.warn_only
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReproducibilityConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Set random seed to {seed}")

def configure_reproducibility(config: ReproducibilityConfig):
    """Configure PyTorch and other libraries for reproducibility.
    
    Args:
        config: Reproducibility configuration
    """
    # Set random seeds
    set_seed(config.seed)
    
    # Configure PyTorch
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        
        if config.use_deterministic_algorithms:
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=config.warn_only)
    
    # Configure environment variables
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    
    # Log configuration
    logger.info("Reproducibility configuration:")
    for k, v in config.to_dict().items():
        logger.info(f"  {k}: {v}")

def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash for reproducibility."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {str(e)}")
        return None

class ExperimentTracker:
    """Track experiment details for reproducibility."""
    
    def __init__(self, output_dir: Union[str, Path], config: Dict[str, Any]):
        """Initialize experiment tracker.
        
        Args:
            output_dir: Directory to save experiment artifacts
            config: Experiment configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.start_time = None
        self.metrics = {}
        
    def start(self):
        """Start tracking the experiment."""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting experiment at {self.start_time.isoformat()}")
        
        # Save initial configuration
        self._save_config()
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics for the experiment.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (e.g., epoch)
        """
        timestamp = datetime.utcnow().isoformat()
        
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                
            self.metrics[name].append({
                'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                'step': step,
                'timestamp': timestamp
            })
        
        # Save metrics after each update
        self._save_metrics()
    
    def save_checkpoint(self, model: torch.nn.Module, filename: str = 'checkpoint.pt'):
        """Save a model checkpoint.
        
        Args:
            model: PyTorch model to save
            filename: Name of the checkpoint file
        """
        checkpoint_path = self.output_dir / filename
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
            'git_commit': get_git_commit_hash(),
            'timestamp': datetime.utcnow().isoformat()
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_config(self):
        """Save experiment configuration to file."""
        config_path = self.output_dir / 'config.json'
        
        # Add metadata to config
        config = {
            **self.config,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'git_commit': get_git_commit_hash(),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else None,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_path = self.output_dir / 'metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.debug(f"Updated metrics in {metrics_path}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        logger.info(f"Experiment completed in {duration:.2f} seconds")
        
        # Save final metrics and configuration
        self._save_metrics()
        
        # Save completion status
        status = {
            'status': 'completed' if exc_type is None else 'failed',
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'exception': str(exc_val) if exc_val else None
        }
        
        status_path = self.output_dir / 'status.json'
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Saved experiment status to {status_path}")
        
        # Don't suppress exceptions
        return False
