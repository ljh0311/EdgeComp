import numpy as np
import tensorflow as tf
import librosa
import joblib
import pyaudio
import threading
import queue
import time
import logging

# Configuration (match with training script)
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
OVERLAP = 1.5  # seconds of overlap between chunks
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP)

class BabyCryMonitor:
    def __init__(self, model_path='baby_cry_classifier_enhanced.tflite', 
                 encoder_path='label_encoder_enhanced.pkl'):
        """
        Initialize baby cry monitor with TFLite model and label encoder
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Load label encoder
        self.label_encoder = joblib.load(encoder_path)
        
        # Audio queue for processing
        self.audio_queue = queue.Queue()
        
        # Stop flag for threading
        self.stop_event = threading.Event()
        
        # PyAudio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
    def extract_advanced_features(self, audio):
        """
        Extract audio features matching training script
        """
        # Pad or truncate to consistent length
        max_length = SAMPLE_RATE * CHUNK_DURATION
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
        
        # Feature extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, 
                                     n_mfcc=20, 
                                     n_fft=2048, 
                                     hop_length=512)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Combine features
        features = np.vstack([
            mfccs, 
            spectral_centroid, 
            spectral_bandwidth, 
            zero_crossing_rate
        ]).T
        
        return features
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for capturing audio stream
        """
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_chunk)
        return (None, pyaudio.paContinue)
    
    def start_recording(self):
        """
        Start continuous audio recording
        """
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=SAMPLE_RATE,
            stream_callback=self.audio_callback
        )
        self.logger.info("Started audio monitoring...")
    
    def process_audio(self):
        """
        Continuously process audio chunks with sliding window
        """
        audio_buffer = np.array([])
        
        while not self.stop_event.is_set():
            # Collect audio chunks
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_buffer = np.concatenate([audio_buffer, chunk])
            
            # Process if we have enough audio
            if len(audio_buffer) >= CHUNK_SIZE:
                current_chunk = audio_buffer[:CHUNK_SIZE]
                
                # Extract features
                features = self.extract_advanced_features(current_chunk)
                features = tf.keras.preprocessing.sequence.pad_sequences(
                    [features], 
                    padding='post', 
                    dtype='float32'
                )
                
                # Classify
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], features)
                self.interpreter.invoke()
                
                predictions = self.interpreter.get_tensor(output_details[0]['index'])
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = self.label_encoder.classes_[predicted_class_index]
                confidence = predictions[0][predicted_class_index]
                
                # Log prediction if confidence is high
                if confidence > 0.7:
                    self.logger.info(f"Detected {predicted_class} (Confidence: {confidence*100:.2f}%)")
                
                # Slide window
                audio_buffer = audio_buffer[int(CHUNK_SIZE - OVERLAP_SIZE):]
            
            time.sleep(0.1)
    
    def run(self):
        """
        Run the baby cry monitor
        """
        # Start recording thread
        recording_thread = threading.Thread(target=self.start_recording)
        recording_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
        
        try:
            # Keep main thread alive
            processing_thread.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
    
    def __del__(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'pyaudio_instance'):
            self.pyaudio_instance.terminate()

def main():
    monitor = BabyCryMonitor()
    monitor.run()

if __name__ == "__main__":
    main()