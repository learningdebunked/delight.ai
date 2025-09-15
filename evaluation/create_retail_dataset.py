from data.retail_dataset import generate_retail_dataset, save_retail_dataset
import pandas as pd

# Create output directory
import os
os.makedirs('data/processed', exist_ok=True)

# Generate and save the dataset
df = generate_retail_dataset(size=1000)
df.to_csv('data/processed/retail_interactions.csv', index=False)
df.to_json('data/processed/retail_interactions.json', orient='records', indent=2)

print("Generated dataset with", len(df), "samples")
print("Sample data:")
print(df[['text', 'region', 'product_category']].head().to_markdown())
