"""
Baby Emotion Recognition Training Script
-------------------------------------
Trains a model to detect baby-specific emotions using HuBERT or Wav2Vec2.
Focuses on emotions relevant to baby monitoring:
- Happy (playful, laughing)
- Sad (crying)
- Hungry (specific cry patterns)
- Uncomfortable (pain, discomfort)
- Tired (sleepy sounds)
- Natural (normal babbling)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BabyEmotionClassifier(nn.Module):
    """HuBERT-based classifier for baby emotions."""
    
    def __init__(self, num_emotions=6):
        super().__init__()
        self.hubert = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, num_emotions)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Get HuBERT features
        features = self.hubert(x).last_hidden_state
        # Average pooling
        pooled = torch.mean(features, dim=1)
        # Classification layers
        x = self.dropout(pooled)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class BabyEmotionDataset(Dataset):
    """Dataset for baby emotion audio data."""
    
    EMOTIONS = {
        'happy': 0,    # Playful, laughing
        'sad': 1,      # Crying
        'hungry': 2,   # Hunger cries
        'pain': 3,     # Discomfort/pain
        'tired': 4,    # Sleepy sounds
        'natural': 5   # Normal babbling
    }
    
    def __init__(self, data_root, metadata_file, feature_extractor):
        self.data_root = Path(data_root)
        self.feature_extractor = feature_extractor
        
        # Load metadata
        with open(metadata_file) as f:
            self.metadata = json.load(f)
            
        # Convert to list for indexing
        self.samples = list(self.metadata.items())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_id, info = self.samples[idx]
        audio_path = self.data_root / info['path']
        
        # Load and preprocess audio
        audio_data = self.load_audio(audio_path)
        features = self.feature_extractor(
            audio_data, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Get emotion label
        emotion_idx = self.EMOTIONS[info['emotion']]
        
        return {
            'input_values': features.input_values.squeeze(),
            'emotion': torch.tensor(emotion_idx)
        }
    
    @staticmethod
    def load_audio(file_path):
        """Load and preprocess audio file."""
        # Implement audio loading logic here
        # Should return numpy array of audio data
        pass

def train_model(config):
    """Train the emotion detection model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = BabyEmotionClassifier(num_emotions=len(BabyEmotionDataset.EMOTIONS))
    model = model.to(device)
    
    # Setup datasets
    train_dataset = BabyEmotionDataset(
        config['data_root'],
        config['train_metadata'],
        feature_extractor
    )
    
    val_dataset = BabyEmotionDataset(
        config['data_root'],
        config['val_metadata'],
        feature_extractor
    )
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_values = batch['input_values'].to(device)
            emotions = batch['emotion'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, emotions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(device)
                emotions = batch['emotion'].to(device)
                
                outputs = model(input_values)
                loss = criterion(outputs, emotions)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += emotions.size(0)
                correct += predicted.eq(emotions).sum().item()
        
        # Log metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        logger.info(f'Epoch {epoch+1}/{config["epochs"]}:')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        logger.info(f'Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(config['output_dir']) / "hubert-base-ls960_emotion.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f'Saved best model to {model_path}')

if __name__ == "__main__":
    # Training configuration
    config = {
        'data_root': 'data/baby_emotions',
        'train_metadata': 'data/train_metadata.json',
        'val_metadata': 'data/val_metadata.json',
        'output_dir': 'models',
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 10
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Train model
    train_model(config) 