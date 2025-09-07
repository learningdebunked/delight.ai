import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Dict, Any, Tuple
import json

class SEDSTrainer:
    """
    Trains models for the SEDS framework.
    """
    
    def __init__(self, data_path: str = '../data/raw/service_interactions.csv'):
        """
        Initialize the trainer with data path.
        
        Args:
            data_path: Path to the training data
        """
        self.data_path = data_path
        self.models = {}
        self.feature_columns = []
        self.target_column = 'satisfaction_score'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the training data.
        
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Convert string representations of dicts to actual dicts
        if 'cultural_profile' in df.columns and isinstance(df['cultural_profile'].iloc[0], str):
            df['cultural_profile'] = df['cultural_profile'].apply(eval)
        if 'emotion_state' in df.columns and isinstance(df['emotion_state'].iloc[0], str):
            df['emotion_state'] = df['emotion_state'].apply(eval)
            
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for training.
        
        Args:
            df: Raw input data
            
        Returns:
            Tuple of (features, target)
        """
        print("Preprocessing data...")
        
        # Extract features from cultural profile
        cultural_features = pd.json_normalize(df['cultural_profile'].tolist())
        cultural_features = cultural_features.add_prefix('cultural_')  # Add prefix to avoid column name conflicts
        
        # Extract features from emotion state
        emotion_features = pd.json_normalize(df['emotion_state'].tolist())
        emotion_features = emotion_features.add_prefix('emotion_')  # Add prefix to avoid column name conflicts
        emotion_features = emotion_features.fillna(0)  # Fill missing emotions with 0
        
        # One-hot encode categorical variables
        categorical_cols = ['region', 'scenario', 'resolution_status']
        categorical_features = pd.get_dummies(df[categorical_cols])
        
        # Combine all features
        X = pd.concat([
            cultural_features,
            emotion_features,
            categorical_features,
            df['duration_seconds'].fillna(0)  # Handle missing duration
        ], axis=1)
        
        # Store feature names for later use
        self.feature_columns = X.columns.tolist()
        
        # Target variable
        y = df[self.target_column]
        
        return X, y
    
    def train_satisfaction_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train a model to predict satisfaction scores.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing model and metrics
        """
        print("Training satisfaction prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained. MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            'model': model,
            'metrics': {
                'mse': float(mse),
                'r2': float(r2)
            },
            'feature_importances': dict(zip(X.columns, model.feature_importances_))
        }
    
    def train_all_models(self):
        """
        Train all models in the SEDS framework.
        """
        # Create models directory if it doesn't exist
        os.makedirs('../../models', exist_ok=True)
        
        # Load and preprocess data
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        # Train satisfaction prediction model
        satisfaction_model = self.train_satisfaction_model(X, y)
        self.models['satisfaction_predictor'] = satisfaction_model
        
        # Save models and metadata
        self.save_models()
        
        return self.models
    
    def save_models(self):
        """Save trained models and metadata."""
        for name, model_data in self.models.items():
            # Save model
            model_path = f"../../models/{name}.joblib"
            joblib.dump(model_data['model'], model_path)
            
            # Save metadata
            metadata = {
                'name': name,
                'metrics': model_data['metrics'],
                'feature_importances': model_data.get('feature_importances', {}),
                'timestamp': pd.Timestamp.now().isoformat(),
                'feature_columns': self.feature_columns
            }
            
            metadata_path = f"../../models/{name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Models and metadata saved to ../../models/")


def main():
    """Main training script."""
    # Initialize trainer
    trainer = SEDSTrainer()
    
    # Train models
    print("Starting model training...")
    models = trainer.train_all_models()
    
    print("\nTraining complete!")
    print("Model performance:")
    for name, model_data in models.items():
        print(f"\n{name}:")
        for metric, value in model_data['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nModels and metadata saved to ../../models/")


if __name__ == "__main__":
    main()
