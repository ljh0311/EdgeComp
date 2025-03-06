import torch
from ultralytics import YOLO
import platform
import time

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set quantization engine based on platform
        if platform.system() == 'Windows':
            torch.backends.quantized.engine = 'onednn'
        elif platform.system() == 'Linux':
            # Check if running on Raspberry Pi
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    if 'Raspberry Pi' in f.read():
                        torch.backends.quantized.engine = 'qnnpack'
            except:
                torch.backends.quantized.engine = 'onednn'
        
        # Load and quantize model
        self.model = YOLO(model_path)
        if self.device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
    
    def detect(self, frame):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.last_time)
            self.last_time = current_time
            
        results = self.model(frame, verbose=False)
        return results[0], self.fps 