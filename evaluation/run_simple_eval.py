import os
import json
import torch
import pandas as pd
import numpy as np

# Set environment variables before importing torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Now import torch with single thread settings
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

# Configuration
MODEL_NAME = 'distilbert-base-uncased'  # Lighter model
BATCH_SIZE = 4  # Reduced batch size
NUM_EPOCHS = 2  # Reduced epochs
LEARNING_RATE = 2e-5

# Enable garbage collection
import gc
gc.enable()

def load_retail_data(filepath):
    """Load retail dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

class RetailDataset(Dataset):
    """Dataset for retail interactions with cultural and emotion annotations."""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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
        
        # Get cultural profile (6 dimensions)
        cultural_profile = torch.tensor([
            item['cultural_profile']['individualism'],
            item['cultural_profile']['power_distance'],
            item['cultural_profile']['uncertainty_avoidance'],
            item['cultural_profile']['masculinity'],
            item['cultural_profile']['long_term_orientation'],
            item['cultural_profile']['indulgence']
        ], dtype=torch.float32)
        
        # Get emotions (5 categories)
        emotions = torch.tensor([
            item['emotions']['happy'],
            item['emotions']['sad'],
            item['emotions']['angry'],
            item['emotions']['surprised'],
            item['emotions']['neutral']
        ], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'cultural_profile': cultural_profile,
            'emotions': emotions,
            'region': item['region'],
            'product_category': item['product_category']
        }

class CulturalModel(nn.Module):
    """Model for cultural dimension prediction."""
    
    def __init__(self, model_name=MODEL_NAME, num_dims=6):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_dims
        )
        # Replace classifier for regression
        self.bert.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_dims),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def train_model(model, train_loader, val_loader, device='cpu', num_epochs=2, lr=2e-5):
    """Train the cultural model with memory optimizations."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Clear memory
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['cultural_profile'].to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Clear variables to free memory
            del input_ids, attention_mask, targets, outputs, loss
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['cultural_profile'].to(device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.logits, targets)
                
                val_loss += loss.item()
                
                # Clear variables to free memory
                del input_ids, attention_mask, targets, outputs, loss
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.MSELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['cultural_profile'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = outputs.logits
            
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    # Calculate MAE for each dimension
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    mae = torch.mean(torch.abs(all_preds - all_targets), dim=0)
    
    return {
        'mse': total_loss / len(test_loader),
        'mae': mae.tolist(),
        'mae_avg': torch.mean(mae).item()
    }

def main():
    # Set device - prefer CPU for stability
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Set deterministic behavior for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load and prepare data
    data_path = 'data/processed/retail_interactions.json'
    os.makedirs('data/processed', exist_ok=True)
    
    if not os.path.exists(data_path):
        print("Generating retail dataset...")
        # Import the function directly from the module
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from evaluation.data.retail_dataset import generate_retail_dataset
        
        df = generate_retail_dataset(size=1000)
        df.to_json(data_path, orient='records', indent=2)
    else:
        df = pd.read_json(data_path)
    
    print(f"Loaded dataset with {len(df)} samples")
    
    # Split data
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = RetailDataset(train_df, tokenizer)
    val_dataset = RetailDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize and train model
    print("Initializing model...")
    model = CulturalModel()
    
    print("Starting training...")
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        device=device, 
        num_epochs=NUM_EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, device=device)
    
    print("\nEvaluation Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE (avg): {metrics['mae_avg']:.4f}")
    print("\nMAE by dimension:")
    dimensions = ['individualism', 'power_distance', 'uncertainty_avoidance', 
                 'masculinity', 'long_term_orientation', 'indulgence']
    for dim, mae in zip(dimensions, metrics['mae']):
        print(f"  {dim}: {mae:.4f}")

if __name__ == "__main__":
    main()
