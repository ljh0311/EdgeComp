"""
Emotion Detection Model Benchmark
==============================
Tool for comparing performance metrics of different emotion detection models.
"""

import time
import psutil
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any
import soundfile as sf
from .models.unified_sound_detector import BaseSoundDetector
from .models.unified_hubert import UnifiedHuBERTDetector
from .models.sound_wav2vec2 import Wav2Vec2EmotionRecognizer
from .models.sound_basic import BasicEmotionRecognizer

class ModelBenchmark:
    """Benchmark different emotion detection models."""
    
    def __init__(self, config: dict):
        """Initialize the benchmark tool.
        
        Args:
            config (dict): Configuration containing model paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.results = {}
        
        # Initialize models
        self.models = {
            'hubert': UnifiedHuBERTDetector(config),
            'wav2vec2': Wav2Vec2EmotionRecognizer(config),
            'basic': BasicEmotionRecognizer(config.get('model_path'))
        }
    
    def load_test_audio(self, audio_path: str) -> np.ndarray:
        """Load test audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Audio data
        """
        audio_data, sr = sf.read(audio_path)
        if sr != 16000:
            # Resample to 16kHz if needed
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        return audio_data
    
    def measure_inference_time(self, model: BaseSoundDetector, audio_data: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
        """Measure inference time statistics.
        
        Args:
            model: Emotion detection model
            audio_data: Audio data to process
            num_runs: Number of inference runs
            
        Returns:
            dict: Inference time statistics
        """
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            model.detect(audio_data)
            times.append(time.time() - start_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def measure_memory_usage(self, model: BaseSoundDetector) -> Dict[str, float]:
        """Measure memory usage of the model.
        
        Returns:
            dict: Memory usage statistics in MB
        """
        process = psutil.Process()
        
        # Get baseline memory
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Run inference to ensure model is loaded
        dummy_audio = np.zeros(16000, dtype=np.float32)
        model.detect(dummy_audio)
        
        # Get peak memory
        peak = process.memory_info().rss / 1024 / 1024
        
        return {
            'baseline': baseline,
            'peak': peak,
            'used': peak - baseline
        }
    
    def measure_gpu_memory(self, model: BaseSoundDetector) -> Dict[str, float]:
        """Measure GPU memory usage if available.
        
        Returns:
            dict: GPU memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {'gpu_memory': 0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Run inference to ensure model is loaded
        dummy_audio = np.zeros(16000, dtype=np.float32)
        model.detect(dummy_audio)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {'gpu_memory': peak_memory}
    
    def benchmark_model(self, model_name: str, audio_data: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive benchmark on a model.
        
        Args:
            model_name: Name of the model to benchmark
            audio_data: Audio data for testing
            
        Returns:
            dict: Benchmark results
        """
        model = self.models[model_name]
        model.start()
        
        results = {
            'model_name': model.model_name,
            'supported_emotions': model.supported_emotions,
            'inference_time': self.measure_inference_time(model, audio_data),
            'memory_usage': self.measure_memory_usage(model),
            'gpu_memory': self.measure_gpu_memory(model)
        }
        
        model.stop()
        return results
    
    def run_benchmarks(self, test_audio_path: str) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on all models.
        
        Args:
            test_audio_path: Path to test audio file
            
        Returns:
            dict: Benchmark results for all models
        """
        try:
            audio_data = self.load_test_audio(test_audio_path)
            
            for model_name in self.models:
                self.logger.info(f"Benchmarking {model_name}...")
                self.results[model_name] = self.benchmark_model(model_name, audio_data)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {str(e)}")
            raise
    
    def print_results(self):
        """Print benchmark results in a formatted way."""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\nEmotion Detection Model Benchmark Results")
        print("========================================")
        
        for model_name, results in self.results.items():
            print(f"\nModel: {results['model_name']}")
            print("-" * (len(results['model_name']) + 7))
            
            print("\nSupported Emotions:")
            print(", ".join(results['supported_emotions']))
            
            print("\nInference Time (seconds):")
            inf_time = results['inference_time']
            print(f"  Mean: {inf_time['mean']:.4f} ± {inf_time['std']:.4f}")
            print(f"  Range: [{inf_time['min']:.4f}, {inf_time['max']:.4f}]")
            
            print("\nMemory Usage (MB):")
            mem = results['memory_usage']
            print(f"  Peak: {mem['peak']:.1f}")
            print(f"  Used: {mem['used']:.1f}")
            
            if results['gpu_memory']['gpu_memory'] > 0:
                print(f"\nGPU Memory (MB): {results['gpu_memory']['gpu_memory']:.1f}")
            
            print("\n" + "="*50)

def main():
    """Run benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark emotion detection models")
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = {
        'model_path': 'models/hubert/',
        'confidence_threshold': 0.5
    }
    
    # Run benchmarks
    benchmark = ModelBenchmark(config)
    benchmark.run_benchmarks(args.audio)
    benchmark.print_results()

if __name__ == "__main__":
    main() 