# Delight.AI Evaluation Framework

This directory contains a comprehensive evaluation framework for the Delight.AI system, designed to ensure rigorous testing and validation of the cultural adaptation and emotion detection capabilities.

## Structure

```
evaluation/
├── baselines/           # Baseline models for comparison
├── data/                # Data loading and preprocessing utilities
├── metrics/             # Custom evaluation metrics
├── notebooks/           # Jupyter notebooks for analysis
├── results/             # Output directory for evaluation results
└── experiment.py        # Main experiment runner
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional requirements for cultural adaptation:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running Experiments

### 1. Training and Evaluation

```python
from evaluation.experiment import Experiment
from evaluation.baselines import CulturalBaseline
from evaluation.data import DataLoader, DatasetConfig

# Initialize experiment
exp = Experiment("cultural_adaptation_eval")

# Load dataset
config = DatasetConfig(
    name="cultural_adaptation",
    path="path/to/dataset.csv",
    text_col="text",
    label_col=["dim1", "dim2", ...],  # Cultural dimensions
    preprocess_fn=preprocess_function
)
data_loader = DataLoader(config)
data = data_loader.load()
train_data, test_data = data_loader.get_splits(test_size=0.2)

# Initialize model
model = CulturalBaseline(
    model_name="bert-base-uncased",
    num_cultural_dims=len(config.label_col)
)

# Run experiment
with exp:
    exp.log_params({
        "model": "bert-base-uncased",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 10
    })
    
    results = exp.run_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=10
    )
```

### 2. Evaluation Only

```python
from evaluation.metrics import MetricCalculator

# Initialize metrics calculator
metrics_calculator = MetricCalculator()

# Compute metrics
metrics = metrics_calculator.compute_cultural_metrics(
    predictions=predictions,
    references=references,
    cultural_dimensions=["individualism", "power_distance", ...]
)

print(metrics)
```

## Available Metrics

### Cultural Adaptation Metrics
- `{dimension}_mae`: Mean Absolute Error for each cultural dimension
- `{dimension}_direction_acc`: Direction accuracy (correct side of neutral)
- `overall_mae`: Average MAE across all dimensions
- `overall_direction_acc`: Average direction accuracy across dimensions

### Emotion Detection Metrics
- `{emotion}_mse`: Mean Squared Error for each emotion
- `{emotion}_corr`: Pearson correlation for each emotion

### Text Generation Metrics
- `rouge1/2/L`: ROUGE scores for text generation
- `bert_score`: BERTScore for semantic similarity

## Baseline Models

### CulturalBaseline
A BERT-based model fine-tuned for predicting cultural dimensions. Outputs values between 0 and 1 for each dimension.

### EmotionBaseline
A BERT-based model for multi-label emotion classification. Uses sigmoid activation for multi-label prediction.

## Best Practices

1. **Reproducibility**:
   - Set random seeds for all libraries (PyTorch, NumPy, Python)
   - Use the `Experiment` class for automatic logging
   - Save model checkpoints and configurations

2. **Evaluation**:
   - Always evaluate on a held-out test set
   - Report both micro and macro averages for imbalanced datasets
   - Include confidence intervals when possible

3. **Documentation**:
   - Document all hyperparameters
   - Save command-line arguments and environment details
   - Include example usage in docstrings

## MLflow Integration

The framework integrates with MLflow for experiment tracking. To use:

1. Start the MLflow UI:
   ```bash
   mlflow ui
   ```

2. Access the dashboard at `http://localhost:5000`

## License

This evaluation framework is part of the Delight.AI project and is licensed under the same terms.
