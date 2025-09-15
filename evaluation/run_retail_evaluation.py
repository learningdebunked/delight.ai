import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Add the evaluation directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baselines import CulturalBaseline, EmotionBaseline
from experiment import Experiment
from metrics import MetricCalculator

class RetailDataset(Dataset):
    """Dataset for retail interactions with cultural and emotion annotations."""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file (CSV or JSON)
            tokenizer: Tokenizer to use for text processing
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        else:  # JSON
            with open(data_path, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
        
        # Convert cultural profiles to tensors
        self.cultural_dims = [
            'individualism', 'power_distance', 'uncertainty_avoidance',
            'masculinity', 'long_term_orientation', 'indulgence'
        ]
        
        # Convert emotions to tensors
        self.emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            str(item['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get cultural profile
        cultural_profile = torch.tensor([
            item['cultural_profile'][dim] for dim in self.cultural_dims
        ], dtype=torch.float32)
        
        # Get emotions
        emotions = torch.tensor([
            item['emotions'][emo] for emo in self.emotions
        ], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'cultural_profile': cultural_profile,
            'emotions': emotions,
            'region': item['region'],
            'product_category': item['product_category']
        }

def evaluate_retail_models(data_path='data/processed/retail_interactions.json'):
    """Run evaluation on the retail dataset."""
    # Initialize tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset and dataloader
    dataset = RetailDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    cultural_model = CulturalBaseline(model_name, num_cultural_dims=len(dataset.cultural_dims))
    emotion_model = EmotionBaseline(model_name, num_emotions=len(dataset.emotions))
    
    # Move models to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cultural_model = cultural_model.to(device)
    emotion_model = emotion_model.to(device)
    
    # Initialize metrics
    metrics_calculator = MetricCalculator()
    
    # Run evaluation
    cultural_preds = []
    emotion_preds = []
    cultural_targets = []
    emotion_targets = []
    
    cultural_model.eval()
    emotion_model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cultural_target = batch['cultural_profile'].to(device)
            emotion_target = batch['emotions'].to(device)
            
            # Get predictions
            cultural_output = cultural_model(input_ids, attention_mask)
            emotion_output = emotion_model(input_ids, attention_mask)
            
            cultural_preds.append(cultural_output.logits.cpu())
            emotion_preds.append(emotion_output.logits.cpu())
            cultural_targets.append(cultural_target.cpu())
            emotion_targets.append(emotion_target.cpu())
    
    # Concatenate all batches
    cultural_preds = torch.cat(cultural_preds, dim=0).numpy()
    emotion_preds = torch.cat(emotion_preds, dim=0).numpy()
    cultural_targets = torch.cat(cultural_targets, dim=0).numpy()
    emotion_targets = torch.cat(emotion_targets, dim=0).numpy()
    
    # Calculate metrics
    cultural_metrics = metrics_calculator.compute_cultural_metrics(
        predictions=[{dim: float(val) for dim, val in zip(dataset.cultural_dims, pred)} 
                    for pred in cultural_preds],
        references=[{dim: float(val) for dim, val in zip(dataset.cultural_dims, target)} 
                   for target in cultural_targets],
        cultural_dimensions=dataset.cultural_dims
    )
    
    emotion_metrics = metrics_calculator.compute_emotion_metrics(
        predictions=[{emo: float(val) for emo, val in zip(dataset.emotions, pred)} 
                    for pred in emotion_preds],
        references=[{emo: float(val) for emo, val in zip(dataset.emotions, target)} 
                   for target in emotion_targets]
    )
    
    return {
        'cultural_metrics': cultural_metrics,
        'emotion_metrics': emotion_metrics,
        'sample_size': len(dataset)
    }

if __name__ == "__main__":
    # Create dataset if it doesn't exist
    if not os.path.exists('data/processed/retail_interactions.json'):
        print("Generating retail dataset...")
        from create_retail_dataset import generate_retail_dataset
        os.makedirs('data/processed', exist_ok=True)
        df = generate_retail_dataset(size=1000)
        df.to_json('data/processed/retail_interactions.json', orient='records', indent=2)
    
    # Run evaluation
    print("Running evaluation on retail dataset...")
    results = evaluate_retail_models()
    
    # Print results
    print("\n=== Cultural Adaptation Metrics ===")
    for k, v in results['cultural_metrics'].items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Emotion Detection Metrics ===")
    for k, v in results['emotion_metrics'].items():
        print(f"{k}: {v:.4f}")
    
    print(f"\nEvaluated on {results['sample_size']} samples.")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/retail_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/retail_evaluation.json")
