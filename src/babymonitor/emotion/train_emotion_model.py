"""
Train Emotion Recognition Model
=============================
Fine-tunes a Wav2Vec2 model on the IEMOCAP dataset for emotion recognition.
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import AdamW, get_linear_schedule_with_warmup
import torchaudio
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """Custom dataset for emotion recognition."""
    
    def __init__(self, tsv_file, audio_dir, processor, max_length=16000):
        """Initialize dataset."""
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Simple emotion label mapping
        self.label2id = {
            'angry': 0, 'happy': 1, 
            'neutral': 2, 'sad': 3
        }
        
        # Check for unknown labels
        for label in self.df['label'].str.lower().unique():
            if label not in self.label2id:
                logger.warning(f"Found unknown label: {label}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Append .wav extension to the filename
        audio_path = self.audio_dir / f"{row['filename']}.wav"
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to numpy array
            waveform = waveform.squeeze().numpy()
            
            # Pad or truncate to max_length
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                padding = np.zeros(self.max_length - len(waveform))
                waveform = np.concatenate([waveform, padding])
            
            # Process audio
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Get label
            label = self.label2id[row['label'].lower()]
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'label': torch.tensor(label)
            }
        except Exception as e:
            logger.error(f"Error processing file {audio_path}: {str(e)}")
            raise

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    learning_rate=2e-5,
    warmup_steps=1000
):
    """Train the model."""
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0
    best_model_path = models_dir / 'best_emotion_model.pt'
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_values=input_values, labels=labels)
                val_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch + 1}:')
        logger.info(f'  Average training loss: {avg_train_loss:.4f}')
        logger.info(f'  Average validation loss: {avg_val_loss:.4f}')
        logger.info(f'  Validation accuracy: {accuracy:.4f}')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'  New best model saved with accuracy: {accuracy:.4f}')

def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize Wav2Vec2 model and processor
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,  # angry, happy, neutral, sad
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Create datasets
    tsv_file = 'data/IEMOCAP_4.tsv'
    audio_dir = 'data/IEMOCAP_full_release_audio'
    
    # Split data
    df = pd.read_csv(tsv_file, sep='\t')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create temporary train/val TSV files
    train_df.to_csv('data/train.tsv', sep='\t', index=False)
    val_df.to_csv('data/val.tsv', sep='\t', index=False)
    
    # Create datasets and dataloaders
    train_dataset = EmotionDataset('data/train.tsv', audio_dir, processor)
    val_dataset = EmotionDataset('data/val.tsv', audio_dir, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    # Clean up temporary files
    os.remove('data/train.tsv')
    os.remove('data/val.tsv')
    
    logger.info('Training completed!')

if __name__ == '__main__':
    main() 