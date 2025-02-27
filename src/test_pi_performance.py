"""
Performance Testing Script for Raspberry Pi
----------------------------------------
Tests the real-time emotion recognition system's performance on Raspberry Pi,
measuring CPU usage, memory consumption, latency, and inference speed.
"""

import os
import time
import psutil
import numpy as np
import threading
from datetime import datetime
from pathlib import Path
import argparse
import json
from typing import Dict, List
import logging
from realtime_emotion import create_recognizer, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pi_performance_test.log'),
        logging.StreamHandler()
    ]
)

class PerformanceMetrics:
    """Tracks and records performance metrics"""
    
    def __init__(self, test_duration: int = 60):
        self.test_duration = test_duration
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.temperature: List[float] = []
        self.latencies: List[float] = []
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring system metrics"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring system metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def add_latency(self, latency: float):
        """Add processing latency measurement"""
        self.latencies.append(latency)
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring and (time.time() - self.start_time) < self.test_duration:
            # CPU usage (percentage)
            self.cpu_usage.append(psutil.cpu_percent(interval=1))
            
            # Memory usage (MB)
            self.memory_usage.append(process.memory_info().rss / 1024 / 1024)
            
            # CPU temperature (Raspberry Pi specific)
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read().strip()) / 1000
                self.temperature.append(temp)
            except:
                self.temperature.append(0)
                
            time.sleep(1)
            
    def get_summary(self) -> Dict:
        """Generate summary statistics"""
        return {
            "cpu_usage": {
                "mean": np.mean(self.cpu_usage),
                "max": np.max(self.cpu_usage),
                "min": np.min(self.cpu_usage)
            },
            "memory_usage_mb": {
                "mean": np.mean(self.memory_usage),
                "max": np.max(self.memory_usage),
                "min": np.min(self.memory_usage)
            },
            "temperature_c": {
                "mean": np.mean(self.temperature),
                "max": np.max(self.temperature),
                "min": np.min(self.temperature)
            },
            "latency_ms": {
                "mean": np.mean(self.latencies) * 1000 if self.latencies else 0,
                "max": np.max(self.latencies) * 1000 if self.latencies else 0,
                "min": np.min(self.latencies) * 1000 if self.latencies else 0
            }
        }

class PerformanceTester:
    """Tests emotion recognition system performance"""
    
    def __init__(self, model_path: str, test_duration: int = 60):
        self.model_path = model_path
        self.test_duration = test_duration
        self.metrics = PerformanceMetrics(test_duration)
        self.recognizer = None
        
    def _process_callback(self, audio_data: np.ndarray, predictions: Dict[str, float], 
                         processing_time: float):
        """Callback for processing results"""
        self.metrics.add_latency(processing_time)
        
    def run_test(self):
        """Run the performance test"""
        try:
            logging.info("Starting performance test...")
            logging.info(f"Model path: {self.model_path}")
            logging.info(f"Test duration: {self.test_duration} seconds")
            
            # Create audio config with smaller block size for testing
            audio_config = AudioConfig(
                sample_rate=16000,
                channels=1,
                block_size=2000,  # 125ms blocks for more frequent measurements
                dtype=np.float32
            )
            
            # Initialize recognizer
            self.recognizer = create_recognizer(self.model_path)
            
            # Start monitoring
            self.metrics.start_monitoring()
            
            # Start recognition
            start_time = time.time()
            self.recognizer.start()
            
            # Run for specified duration
            while (time.time() - start_time) < self.test_duration:
                time.sleep(0.1)
                
            # Stop everything
            self.recognizer.stop()
            self.metrics.stop_monitoring()
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            logging.error(f"Test failed: {str(e)}")
            raise
            
    def _generate_report(self):
        """Generate and save performance report"""
        summary = self.metrics.get_summary()
        
        # Create report directory
        report_dir = Path("performance_reports")
        report_dir.mkdir(exist_ok=True)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"pi_performance_{timestamp}.json"
        
        report = {
            "test_info": {
                "timestamp": timestamp,
                "duration": self.test_duration,
                "model_path": str(self.model_path),
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
                    "platform": "Raspberry Pi"
                }
            },
            "metrics": summary
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Log summary
        logging.info("\nPerformance Test Results:")
        logging.info(f"Average CPU Usage: {summary['cpu_usage']['mean']:.1f}%")
        logging.info(f"Average Memory Usage: {summary['memory_usage_mb']['mean']:.1f} MB")
        logging.info(f"Average Temperature: {summary['temperature_c']['mean']:.1f}°C")
        logging.info(f"Average Latency: {summary['latency_ms']['mean']:.1f} ms")
        logging.info(f"\nFull report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Test emotion recognition performance on Raspberry Pi')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the emotion recognition model')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds (default: 60)')
    args = parser.parse_args()
    
    # Verify model path
    model_path = Path(args.model)
    if not model_path.exists():
        logging.error(f"Model not found at {model_path}")
        return 1
        
    # Run test
    tester = PerformanceTester(str(model_path), args.duration)
    try:
        tester.run_test()
        return 0
    except Exception as e:
        logging.error(f"Test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main()) 