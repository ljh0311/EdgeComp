"""
Create Emotion Recognition Model
============================
Script to create and save a compatible emotion recognition model.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path

class EmotionModel(nn.Module):
    """Emotion recognition model architecture"""
    def __init__(self, input_size=768, hidden_size=256, num_emotions=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_emotions)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_model():
    """Create and initialize the emotion recognition model"""
    model = EmotionModel()
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    return model

def main():
    # Create model
    model = create_model()
    
    # Set model to eval mode
    model.eval()
    
    # Get the models directory
    current_dir = Path(__file__).parent.parent.parent
    models_dir = current_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    save_path = models_dir / "emotion_model.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'input_size': 768,
            'hidden_size': 256,
            'num_emotions': 4
        }
    }, save_path)
    
    print(f"Created and saved emotion model to {save_path}")

if __name__ == "__main__":
    main() 