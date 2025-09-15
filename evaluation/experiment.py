"""Experiment runner for evaluating models."""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DataLoader as CustomDataLoader, DatasetConfig
from .metrics import MetricCalculator

class Experiment:
    """Class to run and track experiments."""
    
    def __init__(
        self,
        name: str,
        output_dir: Union[str, Path] = "results",
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = "http://localhost:5000",
        device: str = None
    ):
        """Initialize the experiment.
        
        Args:
            name: Name of the experiment
            output_dir: Directory to save results
            use_mlflow: Whether to use MLflow for experiment tracking
            mlflow_tracking_uri: URI for MLflow tracking server
            device: Device to run experiments on (cuda, mps, cpu)
        """
        self.name = name
        self.output_dir = Path(output_dir) / name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Set up MLflow if enabled
        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(name)
            self.run = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Set device
        self.device = device or self._get_available_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricCalculator()
    
    def _get_available_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow and local file."""
        if self.use_mlflow:
            mlflow.log_params(params)
        
        # Save to local file
        with open(self.output_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics to MLflow and local file."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        # Save to local file
        metrics_file = self.output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.update(metrics)
            metrics = existing_metrics
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def log_artifacts(self, local_dir: Union[str, Path], artifact_path: str = None) -> None:
        """Log artifacts to MLflow and copy to local directory."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            self.logger.warning(f"Artifact directory {local_dir} does not exist")
            return
            
        if self.use_mlflow:
            mlflow.log_artifacts(local_dir, artifact_path)
        
        # Copy to experiment directory
        dest_dir = self.output_dir / (artifact_path or "")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for item in local_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_dir / item.name)
            else:
                shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
    
    def evaluate_model(
        self,
        model: Any,
        data_loader: DataLoader,
        metrics: List[str] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            metrics: List of metrics to compute
            prefix: Prefix for metric names (e.g., 'val_', 'test_')
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {prefix}"):
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Get predictions and targets
                preds = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs
                all_preds.extend(preds.cpu().numpy())
                
                if 'labels' in batch:
                    all_targets.extend(batch['labels'].cpu().numpy())
        
        # Compute metrics
        results = {}
        
        if 'accuracy' in metrics:
            results[f"{prefix}accuracy"] = accuracy_score(all_targets, all_preds)
        
        if 'f1' in metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            results.update({
                f"{prefix}precision": precision,
                f"{prefix}recall": recall,
                f"{prefix}f1": f1,
            })
        
        # Log metrics
        self.log_metrics(results)
        return results
    
    def run_experiment(
        self,
        model: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        num_epochs: int = 10,
        optimizer_params: Dict[str, Any] = None,
        lr_scheduler_params: Dict[str, Any] = None,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, Any]:
        """Run a complete training and evaluation experiment.
        
        Args:
            model: Model to train and evaluate
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: Optional DataLoader for test data
            num_epochs: Number of training epochs
            optimizer_params: Parameters for the optimizer
            lr_scheduler_params: Parameters for the learning rate scheduler
            early_stopping_patience: Number of epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary with training history and evaluation results
        """
        # Set up optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **(optimizer_params or {'lr': 2e-5})
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **(lr_scheduler_params or {'mode': 'max', 'patience': 2})
        )
        
        # Training loop
        best_val_score = -float('inf')
        patience_counter = 0
        history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                # Move batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_metrics = self.evaluate_model(model, val_loader, metrics=['accuracy', 'f1'], prefix='val_')
            
            # Update learning rate
            scheduler.step(val_metrics['val_f1'])
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                **val_metrics
            }
            
            self.log_metrics(epoch_metrics, step=epoch + 1)
            history.append(epoch_metrics)
            
            # Check for early stopping
            if val_metrics['val_f1'] > best_val_score:
                best_val_score = val_metrics['val_f1']
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_f1': best_val_score,
                    }, f"{checkpoint_dir}/best_model.pt")
                    
                    # Log model artifact
                    if self.use_mlflow:
                        mlflow.pytorch.log_model(model, "best_model")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Final evaluation on test set if provided
        test_metrics = {}
        if test_loader is not None:
            # Load best model
            if checkpoint_dir and os.path.exists(f"{checkpoint_dir}/best_model.pt"):
                checkpoint = torch.load(f"{checkpoint_dir}/best_model.pt")
                model.load_state_dict(checkpoint['model_state_dict'])
            
            test_metrics = self.evaluate_model(model, test_loader, metrics=['accuracy', 'f1'], prefix='test_')
            self.log_metrics(test_metrics)
        
        # Log artifacts
        self.log_artifacts(checkpoint_dir or ".", "checkpoints")
        
        return {
            'history': history,
            'best_val_score': best_val_score,
            'test_metrics': test_metrics
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_mlflow and hasattr(self, 'run'):
            mlflow.end_run()
