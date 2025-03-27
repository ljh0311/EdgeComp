import requests
import json
import os
import sys
import platform
import psutil
from flask import Flask, jsonify, request
from datetime import datetime, timedelta

# Test if the API server is running
def test_server():
    try:
        r = requests.get('http://localhost:5000/')
        print(f"Server status: {'ONLINE' if r.status_code == 200 else 'RESPONDING BUT WITH ISSUES'}")
    except Exception as e:
        print(f"Server status: OFFLINE or UNREACHABLE")
        print(f"Error: {e}")
    
    print("\nTesting API endpoints:")
    test_endpoint('Models API', 'http://localhost:5000/api/emotion/models')
    test_endpoint('Microphones API', 'http://localhost:5000/api/audio/microphones')
    test_endpoint('System Info API', 'http://localhost:5000/api/system/info')

def test_endpoint(name, url):
    try:
        r = requests.get(url)
        print(f"{name}: {'SUCCESS' if r.status_code == 200 else 'FAILED'} (Status code: {r.status_code})")
        if r.status_code == 200:
            try:
                response_data = r.json()
                print(f"  Response data: {json.dumps(response_data, indent=2)[:200]}...")
            except:
                print("  Could not parse JSON response")
        else:
            print(f"  Response text: {r.text[:100]}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")

# Example implementations for the missing API endpoints
def example_implementations():
    print("\nExample implementations for these endpoints:\n")
    
    print("1. MODELS API ENDPOINT (app.py or routes.py):")
    print("""
@app.route('/api/emotion/models', methods=['GET'])
def get_emotion_models():
    # This should be replaced with your actual models data
    models = [
        {
            "id": "basic_1",
            "name": "Basic Emotion Model",
            "type": "basic",
            "emotions": ["crying", "laughing", "babbling", "silence"],
            "is_available": True
        },
        {
            "id": "advanced_1",
            "name": "Advanced Emotion Model",
            "type": "advanced",
            "emotions": ["crying", "laughing", "babbling", "silence", "hungry", "uncomfortable"],
            "is_available": True
        }
    ]
    
    # You might want to determine which model is currently in use
    current_model = models[0]  # Just an example
    
    return jsonify({
        "models": models,
        "current_model": current_model
    })
""")

    print("\n2. MICROPHONES API ENDPOINT (app.py or routes.py):")
    print("""
@app.route('/api/audio/microphones', methods=['GET'])
def get_microphones():
    # This should be replaced with code to detect actual microphones
    # Example using PyAudio to list available audio devices:
    # import pyaudio
    # p = pyaudio.PyAudio()
    # microphones = []
    # for i in range(p.get_device_count()):
    #     device_info = p.get_device_info_by_index(i)
    #     if device_info['maxInputChannels'] > 0:
    #         microphones.append({
    #             "id": str(i),
    #             "name": device_info['name'],
    #             "is_available": True,
    #             "sample_rate": int(device_info['defaultSampleRate']),
    #             "channels": device_info['maxInputChannels']
    #         })
    # p.terminate()
    
    # Fallback example if PyAudio is not available
    microphones = [
        {
            "id": "default",
            "name": "Default System Microphone",
            "is_available": True,
            "sample_rate": 44100,
            "quality": "medium"
        },
        {
            "id": "mic_1",
            "name": "Built-in Microphone",
            "is_available": True,
            "sample_rate": 48000,
            "quality": "medium"
        }
    ]
    
    # Determine which microphone is currently in use
    current_microphone = microphones[0]  # Just an example
    
    return jsonify({
        "microphones": microphones,
        "current_microphone": current_microphone
    })
""")

    print("\n3. SYSTEM INFO API ENDPOINT (app.py or routes.py):")
    print("""
@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    # Get system information
    system_info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__ if 'cv2' in sys.modules else "Not installed",
        "camera_resolution": "640x480",  # Replace with actual resolution
        "camera_device": "0",            # Replace with actual device
        "microphone_device": "Default",  # Replace with actual device
        "microphone_index": 0,           # Replace with actual index
        "audio_sample_rate": 44100,      # Replace with actual sample rate
        "audio_channels": 1,             # Replace with actual channels
        "audio_bit_depth": 16,           # Replace with actual bit depth
        "audio_gain": 0                  # Replace with actual gain
    }
    
    # Get CPU and memory usage
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    memory_available = round(memory.available / (1024 * 1024))
    
    # Get disk usage
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    
    # Calculate uptime
    # This would need to be calculated based on when your application started
    uptime = "01:23:45"  # Replace with actual uptime
    
    # Get model info (you might want to get this from your model manager)
    model_info = {
        "name": "Basic Emotion Model",
        "type": "basic",
        "emotions": ["crying", "laughing", "babbling", "silence"]
    }
    
    return jsonify({
        "system_info": system_info,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "memory_available": memory_available,
        "disk_usage": disk_usage,
        "uptime": uptime,
        "model_info": model_info
    })
""")

    # Create a simple app to show how to implement these
    print("\nTo test these endpoints, you can create a simple Flask app with the following code:")
    print("""
from flask import Flask, jsonify, request
import platform
import sys
import psutil
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "running"})

# Add the three endpoint implementations here

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
""")

if __name__ == '__main__':
    print("BABY MONITOR API TEST UTILITY")
    print("=============================\n")
    test_server()
    print("\n=============================\n")
    example_implementations() 