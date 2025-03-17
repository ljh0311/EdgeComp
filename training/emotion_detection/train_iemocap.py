"""
IEMOCAP Emotion Recognition Training Script
----------------------------------------
Trains a model to detect emotions using the IEMOCAP dataset.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torchaudio
import json
import scipy.signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define emotion mapping
EMOTION_MAP = {
    'N': 0,  # neutral
    'H': 1,  # happy
    'S': 2,  # sad
    'A': 3,  # angry
    'F': 4,  # frustrated
    'E': 5   # excited
}

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, x):
        # Extract wav2vec2 features
        outputs = self.wav2vec2(x)
        # Use mean pooling
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        # Classify
        return self.classifier(pooled)

class IEMOCAPDataset(Dataset):
    """Dataset for IEMOCAP emotion audio data."""
    
    def __init__(self, tsv_file, audio_dir, processor, max_length=16000*5):
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.max_length = max_length
        self.emotion_map = EMOTION_MAP
        
    def __len__(self):
        return len(self.df)
    
    def pad_or_truncate(self, array, target_length):
        """Pad or truncate array to target length."""
        current_length = len(array)
        if current_length > target_length:
            return array[:target_length]
        elif current_length < target_length:
            return np.pad(array, (0, target_length - current_length))
        return array
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.audio_dir / f"{row['filename']}.wav"
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy array
            audio_data = waveform.squeeze().numpy()
            
            # Resample if needed
            if sample_rate != 16000:
                time_steps = len(audio_data)
                new_time_steps = int(time_steps * 16000 / sample_rate)
                audio_data = scipy.signal.resample(audio_data, new_time_steps)
            
            # Pad or truncate to fixed length
            audio_data = self.pad_or_truncate(audio_data, self.max_length)
            
            # Process audio
            inputs = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'emotion': torch.tensor(self.emotion_map[row['label']])
            }
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            # Return a zero tensor of the correct shape as a fallback
            return {
                'input_values': torch.zeros(self.max_length),
                'emotion': torch.tensor(self.emotion_map[row['label']])
            }

def train_model(config):
    """Train the emotion recognition model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = EmotionClassifier(num_emotions=len(EMOTION_MAP))
    model = model.to(device)
    
    # Create datasets
    dataset = IEMOCAPDataset(
        config['tsv_file'],
        config['audio_dir'],
        processor
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
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
            model_path = Path(config['output_dir']) / "wav2vec2_emotion.pt"
            
            # Save the full model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'emotion_map': EMOTION_MAP
            }, model_path)
            
            logger.info(f'Saved best model to {model_path}')

if __name__ == "__main__":
    # Training configuration
    config = {
        'tsv_file': '../../data/IEMOCAP_4.tsv',
        'audio_dir': '../../data/IEMOCAP_full_release_audio',
        'output_dir': '../../models',
        'batch_size': 16,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 10
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Train model
    train_model(config) 