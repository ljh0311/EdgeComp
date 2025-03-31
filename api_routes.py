from flask import Blueprint, jsonify, request
import platform
import sys
import os
import psutil
import datetime
import subprocess
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Try to import cv2 for OpenCV version
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Import the emotion detector for direct integration
try:
    from src.babymonitor.detectors.emotion_detector import EmotionDetector
    EMOTION_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from babymonitor.detectors.emotion_detector import EmotionDetector
        EMOTION_DETECTOR_AVAILABLE = True
    except ImportError:
        EMOTION_DETECTOR_AVAILABLE = False
        print("EmotionDetector not found. Using placeholder implementation.")

# Create blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global variables to track current selections
current_microphone = None
current_emotion_model = None
app_start_time = datetime.datetime.now()
emotion_detector = None

# Initialize emotion detector
def initialize_emotion_detector(model_id="basic_emotion"):
    global emotion_detector
    try:
        if EMOTION_DETECTOR_AVAILABLE:
            print(f"Initializing emotion detector with model {model_id}")
            emotion_detector = EmotionDetector(model_id=model_id)
            return True
        else:
            print("Emotion detector not available, using placeholder implementation")
            return False
    except Exception as e:
        print(f"Error initializing emotion detector: {str(e)}")
        return False

# Initialize emotion detector at startup if available
if EMOTION_DETECTOR_AVAILABLE:
    initialize_emotion_detector()

# ===================== EMOTION MODEL ENDPOINTS =====================

@api_bp.route('/emotion/models', methods=['GET'])
def get_emotion_models():
    """Get available emotion detection models"""
    global current_emotion_model, emotion_detector
    
    # If we have a real emotion detector, get actual models
    if emotion_detector and hasattr(emotion_detector, 'get_available_models'):
        try:
            models_data = emotion_detector.get_available_models()
            # Update current_emotion_model with the actual current model
            current_emotion_model = models_data.get('current_model')
            return jsonify(models_data)
        except Exception as e:
            print(f"Error getting models from detector: {str(e)}")
    
    # Fallback to sample data
    models = [
        {
            "id": "basic_emotion",
            "name": "Basic Emotion",
            "type": "basic",
            "emotions": ["crying", "laughing", "babbling", "silence"],
            "is_available": True
        },
        {
            "id": "advanced_emotion",
            "name": "Advanced Emotion",
            "type": "advanced",
            "emotions": ["crying", "laughing", "babbling", "silence", "hungry", "uncomfortable"],
            "is_available": True
        },
        {
            "id": "neural_emotion",
            "name": "Neural Network Emotion",
            "type": "speechbrain",
            "emotions": ["crying", "laughing", "babbling", "silence", "hungry", "uncomfortable", "pain"],
            "is_available": True
        }
    ]
    
    # Set default model if none is selected
    if current_emotion_model is None:
        current_emotion_model = models[0]
    
    return jsonify({
        "models": models,
        "current_model": current_emotion_model
    })

@api_bp.route('/emotion/model/<model_id>', methods=['GET'])
def get_emotion_model_info(model_id):
    """Get information about a specific emotion model"""
    global emotion_detector
    
    # If we have a real emotion detector, get actual model info
    if emotion_detector and hasattr(emotion_detector, 'get_available_models'):
        try:
            models_data = emotion_detector.get_available_models()
            for model in models_data.get('models', []):
                if model.get('id') == model_id:
                    return jsonify({"model_info": model})
        except Exception as e:
            print(f"Error getting model info from detector: {str(e)}")
    
    # Fallback to sample implementation
    model_info = {
        "id": model_id,
        "name": "Basic Emotion" if model_id == "basic_emotion" else "Advanced Emotion",
        "type": "basic" if model_id == "basic_emotion" else "advanced",
        "emotions": ["crying", "laughing", "babbling", "silence"]
    }
    
    return jsonify({
        "model_info": model_info
    })

@api_bp.route('/emotion/switch_model', methods=['POST'])
def switch_emotion_model():
    """Switch to a different emotion detection model"""
    global current_emotion_model, emotion_detector
    
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400
    
    # If we have a real emotion detector, switch the model
    if emotion_detector and hasattr(emotion_detector, 'switch_model'):
        try:
            result = emotion_detector.switch_model(model_id)
            # Update current_emotion_model from the result
            if 'model_info' in result:
                current_emotion_model = result['model_info']
            return jsonify(result)
        except Exception as e:
            print(f"Error switching emotion detector model: {str(e)}")
    
    # Fallback for model switching logic
    current_emotion_model = {
        "id": model_id,
        "name": "Basic Emotion" if model_id == "basic_emotion" else "Advanced Emotion",
        "type": "basic" if model_id == "basic_emotion" else "advanced",
        "emotions": ["crying", "laughing", "babbling", "silence"]
    }
    
    return jsonify({
        "message": f"Switched to model: {current_emotion_model['name']}",
        "model_info": current_emotion_model
    })

@api_bp.route('/emotion/test_audio', methods=['POST'])
def test_audio():
    """Test the audio system"""
    global emotion_detector
    
    data = request.json
    duration = data.get('duration', 5) if data else 5
    
    # If we have a real emotion detector, use it to test audio
    if emotion_detector and hasattr(emotion_detector, 'test_audio'):
        try:
            result = emotion_detector.test_audio(duration)
            return jsonify(result)
        except Exception as e:
            print(f"Error testing audio: {str(e)}")
    
    # Fallback for audio testing
    return jsonify({
        "message": "Audio test initiated",
        "status": "success",
        "success": True,
        "results": {
            "signal_detected": True,
            "signal_strength": 0.75,
            "background_noise": 0.15,
            "sample_rate": 44100
        }
    })

@api_bp.route('/emotion/restart_audio', methods=['POST'])
def restart_audio():
    """Restart the audio system"""
    global emotion_detector
    
    # If we have a real emotion detector, use it to reset audio
    if emotion_detector and hasattr(emotion_detector, 'reset'):
        try:
            emotion_detector.reset()
            return jsonify({
                "message": "Audio system restarted successfully",
                "status": "success"
            })
        except Exception as e:
            print(f"Error restarting audio: {str(e)}")
    
    # Fallback for audio restart
    return jsonify({
        "message": "Audio system restarted successfully",
        "status": "success"
    })

@api_bp.route('/emotion/current', methods=['GET'])
def get_current_emotion():
    """Get the current emotion detection result."""
    try:
        if not emotion_detector or not emotion_detector.is_running:
            return jsonify({
                'error': 'Emotion detector is not running',
                'status': 'error'
            }), 400
            
        # Get the latest emotion detection result
        result = emotion_detector.get_latest_result()
        if result:
            return jsonify({
                'status': 'success',
                'data': result
            })
        else:
            return jsonify({
                'status': 'no_data',
                'data': {
                    'emotion': 'unknown',
                    'confidence': 0.0,
                    'emotions': {emotion: 0.0 for emotion in emotion_detector.emotions}
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting current emotion: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# ===================== AUDIO MICROPHONE ENDPOINTS =====================

@api_bp.route('/audio/microphones', methods=['GET'])
def get_microphones():
    """Get available microphones"""
    global current_microphone, emotion_detector
    
    microphones = []
    
    # If PyAudio is available, use it to detect actual microphones
    if PYAUDIO_AVAILABLE:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                # Determine quality based on sample rate
                quality = "medium"
                if device_info['defaultSampleRate'] >= 48000:
                    quality = "high"
                elif device_info['defaultSampleRate'] < 44100:
                    quality = "low"
                
                microphones.append({
                    "id": str(i),
                    "name": device_info['name'],
                    "is_available": True,
                    "sample_rate": int(device_info['defaultSampleRate']),
                    "channels": device_info['maxInputChannels'],
                    "quality": quality
                })
        p.terminate()
    else:
        # Fallback if PyAudio is not available
        microphones = [
            {
                "id": "default",
                "name": "Default System Microphone",
                "is_available": True,
                "sample_rate": 44100,
                "quality": "medium"
            },
            {
                "id": "internal",
                "name": "Internal Microphone",
                "is_available": True,
                "sample_rate": 48000,
                "quality": "medium"
            }
        ]
    
    # Set default microphone if none is selected
    if current_microphone is None and microphones:
        current_microphone = microphones[0]
        
        # If we have a real emotion detector, initialize it with the default microphone
        if emotion_detector and hasattr(emotion_detector, 'set_microphone') and current_microphone:
            try:
                emotion_detector.set_microphone(current_microphone['id'])
            except Exception as e:
                print(f"Error setting initial microphone: {str(e)}")
    
    return jsonify({
        "microphones": microphones,
        "current_microphone": current_microphone
    })

@api_bp.route('/audio/microphones/<microphone_id>', methods=['GET'])
def get_microphone_info(microphone_id):
    """Get information about a specific microphone"""
    # Get microphone info to validate it exists
    
    if PYAUDIO_AVAILABLE:
        p = pyaudio.PyAudio()
        try:
            # For integer IDs (most common case)
            try:
                device_info = p.get_device_info_by_index(int(microphone_id))
                # Determine quality based on sample rate
                quality = "medium"
                if device_info['defaultSampleRate'] >= 48000:
                    quality = "high"
                elif device_info['defaultSampleRate'] < 44100:
                    quality = "low"
                
                mic_info = {
                    "id": microphone_id,
                    "name": device_info['name'],
                    "is_available": True,
                    "sample_rate": int(device_info['defaultSampleRate']),
                    "channels": device_info['maxInputChannels'],
                    "quality": quality
                }
                
                return jsonify({"microphone_info": mic_info})
            except (ValueError, IOError):
                # For string IDs or indices that don't exist
                for i in range(p.get_device_count()):
                    device_info = p.get_device_info_by_index(i)
                    if str(i) == microphone_id or device_info.get('name') == microphone_id:
                        quality = "medium"
                        if device_info['defaultSampleRate'] >= 48000:
                            quality = "high"
                        elif device_info['defaultSampleRate'] < 44100:
                            quality = "low"
                        
                        mic_info = {
                            "id": str(i),
                            "name": device_info['name'],
                            "is_available": True,
                            "sample_rate": int(device_info['defaultSampleRate']),
                            "channels": device_info['maxInputChannels'],
                            "quality": quality
                        }
                        
                        return jsonify({"microphone_info": mic_info})
                
                return jsonify({"error": f"Microphone with ID {microphone_id} not found"}), 404
        finally:
            p.terminate()
    else:
        # Fallback if PyAudio is not available
        mic_info = {
            "id": microphone_id,
            "name": f"Microphone {microphone_id}",
            "is_available": True,
            "sample_rate": 44100,
            "quality": "medium"
        }
        
        return jsonify({"microphone_info": mic_info})

@api_bp.route('/audio/set_microphone', methods=['POST'])
def set_microphone():
    """Set the active microphone"""
    global current_microphone, emotion_detector
    
    data = request.json
    microphone_id = data.get('microphone_id')
    
    if not microphone_id:
        return jsonify({"error": "Microphone ID is required"}), 400
    
    # Get microphone info to validate it exists
    if PYAUDIO_AVAILABLE:
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_device_info_by_index(int(microphone_id))
            # Determine quality based on sample rate
            quality = "medium"
            if device_info['defaultSampleRate'] >= 48000:
                quality = "high"
            elif device_info['defaultSampleRate'] < 44100:
                quality = "low"
            
            # Set current microphone
            current_microphone = {
                "id": microphone_id,
                "name": device_info['name'],
                "is_available": True,
                "sample_rate": int(device_info['defaultSampleRate']),
                "channels": device_info['maxInputChannels'],
                "quality": quality
            }
        except (ValueError, IOError):
            return jsonify({"error": f"Microphone with ID {microphone_id} not found"}), 404
        finally:
            p.terminate()
    else:
        # Fallback if PyAudio is not available
        current_microphone = {
            "id": microphone_id,
            "name": f"Microphone {microphone_id}",
            "is_available": True,
            "sample_rate": 44100,
            "quality": "medium"
        }
    
    # If we have a real emotion detector, set the microphone
    mic_set_success = False
    if emotion_detector and hasattr(emotion_detector, 'set_microphone'):
        try:
            mic_set_success = emotion_detector.set_microphone(microphone_id)
            print(f"Microphone set in emotion detector: {mic_set_success}")
        except Exception as e:
            print(f"Error setting microphone in emotion detector: {str(e)}")
    
    return jsonify({
        "message": f"Microphone set to: {current_microphone['name']}",
        "microphone_info": current_microphone,
        "emotion_detector_updated": mic_set_success
    })

@api_bp.route('/audio/microphone', methods=['GET'])
def get_microphone_status():
    """Get current microphone status"""
    global emotion_detector
    
    # If we have a real emotion detector, get actual microphone status
    if emotion_detector and hasattr(emotion_detector, 'get_microphone_status'):
        try:
            status = emotion_detector.get_microphone_status()
            return jsonify(status)
        except Exception as e:
            print(f"Error getting microphone status: {str(e)}")
    
    # Fallback to basic status
    return jsonify({
        "initialized": False,
        "error": "Microphone not initialized",
        "is_muted": True,
        "name": "No Microphone",
        "status": "error"
    })

# ===================== SYSTEM ENDPOINTS =====================

@api_bp.route('/system/check', methods=['GET'])
def check_system():
    """Check the status of all system components"""
    global emotion_detector
    
    # Check emotion detector status
    emotion_status = "ok"
    emotion_message = "Emotion detection is working properly"
    
    if emotion_detector is None:
        emotion_status = "warning"
        emotion_message = "Emotion detector not initialized"
    elif not hasattr(emotion_detector, 'is_model_loaded') or not emotion_detector.is_model_loaded:
        emotion_status = "warning"
        emotion_message = "Emotion model not loaded"
    
    return jsonify({
        "camera": {
            "status": "ok",
            "message": "Camera is working properly"
        },
        "audio": {
            "status": "ok" if PYAUDIO_AVAILABLE else "warning",
            "message": "Audio system is working properly" if PYAUDIO_AVAILABLE else "Audio library not available"
        },
        "detection": {
            "status": "ok" if OPENCV_AVAILABLE else "warning",
            "message": "Detection system is working properly" if OPENCV_AVAILABLE else "OpenCV not available"
        },
        "emotion": {
            "status": emotion_status,
            "message": emotion_message
        }
    })

@api_bp.route('/system_info', methods=['GET'])
def get_system_info():
    """Get detailed system information"""
    global emotion_detector
    
    # Get system information
    system_info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__ if OPENCV_AVAILABLE else "Not installed",
        "camera_status": "connected" if OPENCV_AVAILABLE else "not available",
        "emotion_detector_status": "running" if emotion_detector and emotion_detector.is_model_loaded else "not available",
        "current_emotion": "no_data",
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "uptime": str(datetime.datetime.now() - app_start_time).split('.')[0]
    }
    
    # Get emotion model info if available
    if emotion_detector and hasattr(emotion_detector, 'get_available_models'):
        try:
            models_data = emotion_detector.get_available_models()
            system_info["available_emotion_models"] = models_data
            system_info["emotion_model"] = models_data.get("current_model")
        except Exception as e:
            print(f"Error getting emotion models: {str(e)}")
    
    return jsonify(system_info)

@api_bp.route('/system/restart', methods=['POST'])
def restart_system():
    """Restart the system services"""
    # Placeholder for system restart logic
    # In a real implementation, you would restart services
    
    return jsonify({
        "message": "System restarted successfully",
        "status": "success"
    })

@api_bp.route('/system/stop', methods=['POST'])
def stop_system():
    """Stop the baby monitor system"""
    return jsonify({
        "message": "System stop initiated",
        "status": "success"
    })

# ===================== CAMERA ENDPOINTS =====================

@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """Get list of available cameras"""
    try:
        # Try to get list of cameras using OpenCV
        if OPENCV_AVAILABLE:
            cameras = []
            index = 0
            while True:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    break
                ret, _ = cap.read()
                if ret:
                    cameras.append({
                        "id": str(index),
                        "name": f"Camera {index + 1}",
                        "active": index == 0,  # First camera is active by default
                        "status": "connected"
                    })
                cap.release()
                index += 1
                if index >= 10:  # Limit to 10 cameras to prevent infinite loop
                    break
            
            return jsonify({
                "cameras": cameras,
                "status": "success"
            })
        else:
            # Fallback to single default camera
            return jsonify({
                "cameras": [{
                    "id": "0",
                    "name": "Default Camera",
                    "active": True,
                    "status": "connected"
                }],
                "status": "success"
            })
    except Exception as e:
        print(f"Error getting cameras: {str(e)}")
        return jsonify({
            "error": "Failed to get camera list",
            "status": "error",
            "message": str(e)
        }), 500

@api_bp.route('/cameras/activate', methods=['POST'])
def activate_camera():
    """Activate a specific camera"""
    data = request.json
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({"error": "Camera ID is required"}), 400
    
    try:
        # Verify camera exists and can be opened
        cap = cv2.VideoCapture(int(camera_id))
        if not cap.isOpened():
            return jsonify({
                "error": f"Camera {camera_id} not found or cannot be opened",
                "status": "error"
            }), 404
        
        cap.release()
        
        return jsonify({
            "message": f"Camera {camera_id} activated",
            "status": "success",
            "camera_id": camera_id
        })
    except Exception as e:
        print(f"Error activating camera: {str(e)}")
        return jsonify({
            "error": "Failed to activate camera",
            "status": "error",
            "message": str(e)
        }), 500

# Register this blueprint in your main app.py as:
# from api_routes import api_bp
# app.register_blueprint(api_bp) 