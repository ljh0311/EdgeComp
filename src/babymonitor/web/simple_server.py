"""
Simple Baby Monitor Web Server
============================
A simplified web server for the Baby Monitor System that works reliably on Windows.
"""

import os
import json
import time
import signal
import socket
import threading
import logging
import eventlet

eventlet.monkey_patch()  # Patch standard library for eventlet compatibility
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
from flask_socketio import SocketIO
import numpy as np
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SimpleBabyMonitorWeb")


def find_free_port(start_port=5000, max_port=5100):
    """Find a free port to use"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError("No free ports available")


class SimpleBabyMonitorWeb:
    """
    Simplified web interface for the Baby Monitor System
    """

    def __init__(
        self,
        camera=None,
        person_detector=None,
        emotion_detector=None,
        host="0.0.0.0",
        port=5000,
        debug=False,
        mode="normal",
    ):
        """
        Initialize the web server
        
        Args:
            camera: Camera instance
            person_detector: Person detector instance
            emotion_detector: Emotion detector instance
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
            mode: Web interface mode ('normal' or 'dev')
        """
        self.host = host
        self.port = find_free_port(port)  # Find an available port
        self.debug = debug
        self.mode = mode
        self.camera = camera
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        
        # Create Flask app and Socket.IO
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"),
        )
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", async_mode="eventlet"
        )
        
        # Frame buffer
        self.frame_buffer = None
        self.frame_lock = threading.Lock()

        # Audio level tracking
        self.current_audio_level = -60  # Default to -60 dB (silence)
        self.audio_level_lock = threading.Lock()

        # Thread control
        self.stop_event = threading.Event()
        self.running = False

        # Start time
        self.start_time = time.time()
        
        # Metrics
        self.metrics = {
            "current": {
                "fps": 0,
                "detections": 0,
                "emotion": "unknown",
                "emotion_confidence": 0.0,
            },
            "history": {
                "emotions": {
                    "crying": 0,
                    "laughing": 0,
                    "babbling": 0,
                    "silence": 0,
                    "unknown": 0,
                }
            },
            "detection_types": {"face": 0, "upper_body": 0, "full_body": 0},
            "total_detections": 0,
        }
        
        # Initialize emotion distribution with supported emotions
        if emotion_detector and hasattr(emotion_detector, "emotions"):
            self.metrics["history"]["emotions"] = {
                emotion: 0 for emotion in emotion_detector.emotions
            }
        
        # System status
        self.system_status = {
            "uptime": "00:00:00",
            "cpu_usage": 0,
            "memory_usage": 0,
            "camera_status": "connected" if self.camera else "disconnected",
            "person_detector_status": "running" if self.person_detector else "stopped",
            "emotion_detector_status": (
                "running" if self.emotion_detector else "stopped"
            ),
        }
        
        # Activity log for emotion events
        self.emotion_log = []
        self.max_log_entries = 50  # Keep last 50 entries
        
        # Setup routes
        self._setup_routes()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        
        # Start system status update thread
        self.status_thread = threading.Thread(target=self._update_system_status)
        self.status_thread.daemon = True

        # Start audio level update thread
        self.audio_thread = threading.Thread(target=self._update_audio_level)
        self.audio_thread.daemon = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize alerts list
        self.alerts = []
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route("/")
        def index():
            """Main page"""
            if self.mode == "dev":
                return redirect(url_for("metrics"))
            return render_template(
                "index.html", mode=self.mode, dev_mode=self.mode == "dev"
            )

        @self.app.route("/metrics")
        def metrics():
            """Metrics page"""
            return render_template(
                "metrics.html", mode=self.mode, dev_mode=self.mode == "dev"
            )
        
        @self.app.route("/video_feed")
        def video_feed():
            """Video feed endpoint"""
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/repair/api/audio/level")
        def get_audio_level():
            """API endpoint for current audio level"""
            with self.audio_level_lock:
                # Convert numpy float32 to Python float
                level = float(self.current_audio_level) if self.current_audio_level is not None else -60.0
                return jsonify({"level": level})

        @self.app.route("/api/metrics")
        def api_metrics():
            """API endpoint for metrics"""
            # Get time range from query parameter
            time_range = request.args.get("time_range", "1h")
            
            # Get emotion history from detector if available
            emotion_history = {}
            emotion_percentages = {}
            emotion_timeline = []
            supported_emotions = []
            
            if self.emotion_detector and hasattr(
                self.emotion_detector, "get_emotion_history"
            ):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    emotion_percentages = history_data.get("percentages", {})
                    emotion_timeline = history_data.get("emotions", [])
                    supported_emotions = self.emotion_detector.emotions
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
            else:
                # Use current simple metrics as fallback
                if hasattr(self, "metrics") and "history" in self.metrics:
                    emotion_counts = self.metrics["history"].get("emotions", {})
                    total = sum(emotion_counts.values()) or 1  # Avoid division by zero
                    emotion_percentages = {
                        k: (v / total * 100) for k, v in emotion_counts.items()
                    }
            
            # Enhanced metrics structure to match what the metrics.js expects
            metrics_data = {
                "current": {
                    "fps": self.metrics["current"]["fps"],
                    "detections": self.metrics["current"].get("detections", 0),
                    "emotion": self.metrics["current"].get("emotion", "unknown"),
                    "emotion_confidence": self.metrics["current"].get(
                        "emotion_confidence", 0.0
                    ),
                },
                "history": {"emotions": emotion_percentages},
                "emotion_timeline": emotion_timeline,
                "supported_emotions": supported_emotions,
                "detection_types": self.metrics["detection_types"],
                "total_detections": self.metrics.get("total_detections", 0),
                # Add YOLOv8 specific metrics
                "detection_confidence_avg": 0.85,  # Example value, adjust based on actual data
                "peak_detections": max(
                    1, self.metrics["current"].get("detections", 0)
                ),  # Default to at least 1
                "frame_skip": 2,  # Frame skip rate from person detector
                "process_resolution": "640x480",  # Processing resolution
                "confidence_threshold": 0.5,  # Detection confidence threshold
                "detection_history_size": 5,  # History size for detection smoothing
                "detector_model": "YOLOv8n",  # Detector model name
            }
            
            # If person detector is available, get real values
            if self.person_detector:
                metrics_data["frame_skip"] = getattr(
                    self.person_detector, "frame_skip", 2
                )
                metrics_data["confidence_threshold"] = getattr(
                    self.person_detector, "threshold", 0.5
                )
                metrics_data["detection_history_size"] = getattr(
                    self.person_detector, "max_history_size", 5
                )
                
                # Get process resolution if available
                process_res = getattr(self.person_detector, "process_resolution", None)
                if (
                    process_res
                    and isinstance(process_res, tuple)
                    and len(process_res) == 2
                ):
                    metrics_data["process_resolution"] = (
                        f"{process_res[0]}x{process_res[1]}"
                    )
                    
                # Calculate confidence average from detection history if available
                detection_history = getattr(
                    self.person_detector, "detection_history", []
                )
                if (
                    detection_history
                    and hasattr(self.person_detector, "last_result")
                    and self.person_detector.last_result
                ):
                    detections = self.person_detector.last_result.get("detections", [])
                    if detections:
                        confidence_sum = sum(d.get("confidence", 0) for d in detections)
                        if len(detections) > 0:
                            metrics_data["detection_confidence_avg"] = (
                                confidence_sum / len(detections)
                            )
                            
            # Add model info if emotion detector is available
            if self.emotion_detector:
                metrics_data["emotion_model"] = {
                    "id": getattr(self.emotion_detector, "model_id", "unknown"),
                    "name": getattr(self.emotion_detector, "model_info", {}).get(
                        "name", "Unknown Model"
                    ),
                    "emotions": getattr(self.emotion_detector, "emotions", ["unknown"]),
                }
                
            # Add recent emotion log entries
            metrics_data["emotion_log"] = self.emotion_log[-10:]  # Last 10 entries
            
            return jsonify(metrics_data)
        
        @self.app.route("/api/emotion_history")
        def api_emotion_history():
            """API endpoint for detailed emotion history"""
            time_range = request.args.get("time_range", "1h")
            
            if self.emotion_detector and hasattr(
                self.emotion_detector, "get_emotion_history"
            ):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    return jsonify({"status": "success", "history": history_data})
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
                    return jsonify({"status": "error", "message": str(e)}), 500
            else:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Emotion detector not available or does not support history",
                        }
                    ),
                    404,
                )

        @self.app.route("/api/emotion_log")
        def api_emotion_log():
            """API endpoint for emotion event log"""
            return jsonify({"status": "success", "log": self.emotion_log})
        
        @self.app.route("/api/system_info")
        def api_system_info():
            """API endpoint for system info"""
            info = self.system_status.copy()
            
            # Add emotion model info if available
            if self.emotion_detector:
                info["emotion_model"] = {
                    "id": getattr(self.emotion_detector, "model_id", "unknown"),
                    "name": getattr(self.emotion_detector, "model_info", {}).get(
                        "name", "Unknown Model"
                    ),
                    "emotions": getattr(self.emotion_detector, "emotions", ["unknown"]),
                }
                
                # Add available models if the method exists
                if hasattr(self.emotion_detector, "get_available_models"):
                    try:
                        info["available_emotion_models"] = (
                            self.emotion_detector.get_available_models()
                        )
                    except Exception as e:
                        logger.error(
                            f"Error getting available emotion models: {str(e)}"
                        )
            
            return jsonify(info)
        
        @self.app.route("/repair")
        def repair_tools():
            """Repair tools page"""
            return render_template(
                "repair_tools.html", mode=self.mode, dev_mode=self.mode == "dev"
            )
        
        @self.app.route("/repair/run", methods=["POST"])
        def repair_run():
            """API endpoint for repair tools"""
            try:
                tool = request.form.get("tool")
                if tool == "restart_camera":
                    if self.camera:
                        self.camera.release()
                        time.sleep(1)
                        self.camera.open(0)
                        return jsonify(
                            {
                                "status": "success",
                                "message": "Camera restarted successfully",
                            }
                        )
                    return jsonify(
                        {"status": "error", "message": "Camera not initialized"}
                    )

                elif tool == "restart_audio":
                    if self.emotion_detector:
                        self.emotion_detector.reset()
                        return jsonify(
                            {
                                "status": "success",
                                "message": "Audio system restarted successfully",
                            }
                        )
                    return jsonify(
                        {"status": "error", "message": "Audio system not initialized"}
                    )

                elif tool == "restart_system":
                    if self.camera:
                        self.camera.release()
                        time.sleep(1)
                        self.camera.open(0)
                    if self.emotion_detector:
                        self.emotion_detector.reset()
                    if self.person_detector:
                        self.person_detector.reset()
                    return jsonify(
                        {
                            "status": "success",
                            "message": "System restarted successfully",
                        }
                    )

                elif tool == "switch_emotion_model":
                    model_id = request.form.get("model_id")
                    if not model_id:
                        return jsonify(
                            {"status": "error", "message": "No model ID provided"}
                        )
                        
                    if self.emotion_detector and hasattr(
                        self.emotion_detector, "switch_model"
                    ):
                        try:
                            result = self.emotion_detector.switch_model(model_id)
                            # Update metrics to reflect new emotion set
                            if (
                                "model_info" in result
                                and "emotions" in result["model_info"]
                            ):
                                self.metrics["history"]["emotions"] = {
                                    emotion: 0
                                    for emotion in result["model_info"]["emotions"]
                                }
                                
                            # Emit model change event
                            self.socketio.emit(
                                "emotion_model_changed", result["model_info"]
                            )

                            return jsonify(
                                {
                                    "status": "success",
                                    "message": f"Switched to model: {result['model_info']['name']}",
                                    "model_info": result["model_info"],
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error switching emotion model: {str(e)}")
                            return jsonify({"status": "error", "message": str(e)})
                    return jsonify(
                        {
                            "status": "error",
                            "message": "Emotion detector not initialized or does not support model switching",
                        }
                    )

                return jsonify({"status": "error", "message": "Invalid repair tool"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
        
        # Endpoint for emotion model actions
        @self.app.route("/api/emotion/models", methods=["GET"])
        def get_emotion_models():
            """Get available emotion models"""
            if self.emotion_detector and hasattr(
                self.emotion_detector, "get_available_models"
            ):
                try:
                    models_data = self.emotion_detector.get_available_models()
                    return jsonify(models_data)
                except Exception as e:
                    logger.error(f"Error getting emotion models: {str(e)}")
                    return jsonify({"error": str(e)}), 500
            return (
                jsonify(
                    {
                        "error": "Emotion detector not initialized or does not support model listing"
                    }
                ),
                404,
            )

        @self.app.route("/api/emotion/switch_model", methods=["POST"])
        def switch_emotion_model():
            """Switch to a different emotion model"""
            if not self.emotion_detector or not hasattr(
                self.emotion_detector, "switch_model"
            ):
                return (
                    jsonify(
                        {
                            "error": "Emotion detector not initialized or does not support model switching"
                        }
                    ),
                    404,
                )
                
            try:
                data = request.get_json()
                if not data or "model_id" not in data:
                    return jsonify({"error": "No model ID provided"}), 400
                    
                model_id = data.get("model_id")
                result = self.emotion_detector.switch_model(model_id)
                
                # Update metrics to reflect new emotion set
                if "model_info" in result and "emotions" in result["model_info"]:
                    self.metrics["history"]["emotions"] = {
                        emotion: 0 for emotion in result["model_info"]["emotions"]
                    }
                    
                # Emit model change event
                self.socketio.emit("emotion_model_changed", result["model_info"])
                
                # Log the model change
                self._add_to_emotion_log(
                    {
                        "type": "model_changed",
                        "model": result["model_info"]["name"],
                        "timestamp": time.time(),
                    }
                )

                return jsonify(
                    {
                        "status": "success",
                        "message": f"Switched to model: {result['model_info']['name']}",
                        "model_info": result["model_info"],
                    }
                )
            except Exception as e:
                logger.error(f"Error switching emotion model: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500
        
        # Socket.io events
        @self.socketio.on("request_metrics")
        def handle_request_metrics(data):
            """Handle request for metrics data"""
            time_range = data.get("timeRange", "1h")
            
            # Get emotion history from detector
            if self.emotion_detector and hasattr(
                self.emotion_detector, "get_emotion_history"
            ):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    # Emit history data
                    self.socketio.emit("emotion_history", history_data)
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
        
        if self.mode == "dev":

            @self.app.route("/dev/tools")
            def dev_tools():
                """Developer tools page"""
                return render_template("dev_tools.html", mode=self.mode, dev_mode=True)
            
            @self.app.route("/dev/logs")
            def dev_logs():
                """Logs page"""
                return render_template("logs.html", mode=self.mode, dev_mode=True)

        @self.app.route("/repair/api/cameras")
        def get_cameras():
            """Get available cameras"""
            try:
                cameras = []
                if self.camera:
                    cameras.append(
                        {
                            "id": 0,  # Default camera ID
                            "name": "Default Camera",
                            "active": True,
                            "resolutions": [
                                "640x480",
                                "1280x720",
                                "1920x1080",
                            ],  # Common resolutions
                        }
                    )
                return jsonify({"cameras": cameras})
            except Exception as e:
                logger.error(f"Error getting cameras: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/repair/api/camera/activate", methods=["POST"])
        def activate_camera():
            """Activate a specific camera"""
            try:
                data = request.get_json()
                camera_id = data.get("camera_id")
                if camera_id is not None:
                    # Here you would implement the camera switching logic
                    # For now, we just return success since we only support one camera
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"Activated camera {camera_id}",
                        }
                    )
                return (
                    jsonify({"status": "error", "message": "No camera ID provided"}),
                    400,
                )
            except Exception as e:
                logger.error(f"Error activating camera: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/repair/api/camera/configure", methods=["POST"])
        def configure_camera():
            """Configure camera resolution"""
            try:
                data = request.get_json()
                width = data.get("width")
                height = data.get("height")
                if width and height:
                    # Here you would implement the resolution change logic
                    # For now, we just return success
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"Set resolution to {width}x{height}",
                        }
                    )
                return (
                    jsonify(
                        {"status": "error", "message": "Invalid resolution parameters"}
                    ),
                    400,
                )
            except Exception as e:
                logger.error(f"Error configuring camera: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/repair/api/audio/microphones")
        def get_microphones():
            """Get available microphones"""
            try:
                microphones = []
                current_mic_id = None
                
                # Get current microphone ID from emotion detector if available
                if self.emotion_detector and hasattr(self.emotion_detector, "current_microphone"):
                    current_mic_id = self.emotion_detector.current_microphone

                # Try to get system microphones using sounddevice if available
                try:
                    import sounddevice as sd
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device["max_input_channels"] > 0:  # Input devices only
                            is_current = current_mic_id is not None and str(i) == str(current_mic_id)
                            mic = {
                                "id": i,
                                "name": device["name"],
                                "channels": device["max_input_channels"],
                                "default": device.get("default_input", False),
                                "active": is_current
                            }
                            if is_current:
                                microphones.insert(0, mic)  # Put current mic first
                            else:
                                microphones.append(mic)
                except ImportError:
                    # Fallback to basic microphone detection
                    try:
                        import pyaudio
                        p = pyaudio.PyAudio()
                        for i in range(p.get_device_count()):
                            device_info = p.get_device_info_by_index(i)
                            if device_info["maxInputChannels"] > 0:
                                is_current = current_mic_id is not None and str(i) == str(current_mic_id)
                                mic = {
                                    "id": i,
                                    "name": device_info["name"],
                                    "channels": device_info["maxInputChannels"],
                                    "default": i == p.get_default_input_device_info()["index"],
                                    "active": is_current
                                }
                                if is_current:
                                    microphones.insert(0, mic)  # Put current mic first
                                else:
                                    microphones.append(mic)
                        p.terminate()
                    except ImportError:
                        # If no audio libraries are available, add default microphone
                        microphones.append({
                            "id": 0,
                            "name": "Default System Microphone",
                            "channels": 2,
                            "default": True,
                            "active": True
                        })

                return jsonify({
                    "microphones": microphones,
                    "current_microphone": current_mic_id
                })
            except Exception as e:
                logger.error(f"Error getting microphones: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/repair/api/audio/set_microphone", methods=["POST"])
        def set_microphone():
            """Set active microphone"""
            try:
                data = request.get_json()
                mic_id = data.get("microphone_id")
                if mic_id is not None:
                    # Try to set the microphone in the emotion detector
                    if self.emotion_detector and hasattr(
                        self.emotion_detector, "set_microphone"
                    ):
                        success = self.emotion_detector.set_microphone(str(mic_id))
                        if success:
                            # Wait briefly for the audio system to initialize
                            time.sleep(0.5)
                            # Get the current audio level after switching and convert to Python float
                            current_level = float(self.emotion_detector.get_audio_level())
                            with self.audio_level_lock:
                                self.current_audio_level = current_level

                            return jsonify(
                                {
                                    "status": "success",
                                    "message": f"Set microphone {mic_id}",
                                    "audio_level": current_level,
                                }
                            )
                        else:
                            return (
                                jsonify(
                                    {
                                        "status": "error",
                                        "message": "Failed to set microphone",
                                    }
                                ),
                                500,
                            )
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Emotion detector not available or does not support microphone switching",
                            }
                        ),
                        404,
                    )
                return (
                    jsonify(
                        {"status": "error", "message": "No microphone ID provided"}
                    ),
                    400,
                )
            except Exception as e:
                logger.error(f"Error setting microphone: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/repair/api/system/info")
        def get_system_info():
            """Get system information"""
            try:
                info = {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "camera_status": (
                        "Active"
                        if self.camera and self.camera.is_opened()
                        else "Inactive"
                    ),
                    "audio_status": "Active" if self.emotion_detector else "Inactive",
                }
                return jsonify(info)
            except Exception as e:
                logger.error(f"Error getting system info: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/repair/api/emotion/models")
        def get_repair_emotion_models():
            """Get list of available emotion models for repair tools."""
            try:
                if not self.emotion_detector:
                    return jsonify({
                        "status": "error",
                        "message": "Emotion detector not initialized"
                    }), 404

                models = []
                current_model = None
                model_dirs = ["wav2vec2", "custom", "emotion", "emotion2", "cry_detection"]
                base_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "models",
                    "emotion"
                )

                # Get current model from emotion detector
                if hasattr(self.emotion_detector, "current_model"):
                    current_model = self.emotion_detector.current_model

                # Scan model directories
                for model_dir in model_dirs:
                    dir_path = os.path.join(base_path, model_dir)
                    if os.path.exists(dir_path):
                        for file in os.listdir(dir_path):
                            if file.endswith((".pt", ".pth")):
                                model_name = os.path.splitext(file)[0]
                                model_id = f"{model_dir}/{model_name}"
                                
                                # Determine emotions list based on model type
                                if model_dir == "cry_detection":
                                    emotions = ["cry", "not_cry"]
                                elif model_dir in ["emotion", "emotion2"]:
                                    emotions = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
                                elif model_dir == "wav2vec2":
                                    emotions = ["happy", "sad", "angry", "neutral", "fear"]
                                else:
                                    emotions = ["happy", "sad", "angry", "neutral"]

                                model = {
                                    "id": model_id,
                                    "name": model_name,
                                    "type": model_dir,
                                    "path": os.path.join(model_dir, file),
                                    "description": f"{model_dir.replace('_', ' ').title()} model for emotion detection",
                                    "emotions": emotions,
                                    "active": model_id == current_model
                                }

                                # Put current model first in the list
                                if model["active"]:
                                    models.insert(0, model)
                                else:
                                    models.append(model)

                return jsonify({
                    "status": "success",
                    "models": models,
                    "current_model": current_model
                })

            except Exception as e:
                logger.error(f"Error getting emotion models: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500

        @self.app.route("/repair/api/emotion/switch_model", methods=["POST"])
        def repair_switch_emotion_model():
            """Switch emotion model from repair tools"""
            try:
                data = request.get_json()
                model_id = data.get("model_id")
                if (
                    model_id
                    and self.emotion_detector
                    and hasattr(self.emotion_detector, "switch_model")
                ):
                    result = self.emotion_detector.switch_model(model_id)
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"Switched to model: {model_id}",
                            "model_info": result.get("model_info", {}),
                        }
                    )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Invalid model ID or emotion detector not available",
                        }
                    ),
                    400,
                )
            except Exception as e:
                logger.error(f"Error switching emotion model: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def _generate_frames(self):
        """Generate video frames for streaming"""
        import cv2
        
        # Create a blank frame for when no camera feed is available
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank_frame,
            "No camera feed available",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        _, blank_buffer = cv2.imencode(".jpg", blank_frame)
        blank_bytes = blank_buffer.tobytes()
        
        while True:
            try:
                with self.frame_lock:
                    if self.frame_buffer is None:
                        # If no frame is available, yield a blank frame
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + blank_bytes + b"\r\n"
                        )
                    else:
                        # Yield the frame
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + self.frame_buffer
                            + b"\r\n"
                        )
                
                # Sleep to control frame rate
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error generating frames: {e}")
                import traceback

                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def _add_to_emotion_log(self, entry):
        """Add an entry to the emotion log"""
        self.emotion_log.append(entry)
        # Trim log if it gets too big
        if len(self.emotion_log) > self.max_log_entries:
            self.emotion_log = self.emotion_log[-self.max_log_entries :]
    
    def _process_frames(self):
        """Process frames from camera"""
        import cv2
        
        # Initialize detection history for smoothing
        detection_history = []
        max_history = 5
        last_fps_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.metrics["current"]["fps"] = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                
                # Process frame with person detector
                results = self.person_detector.process_frame(frame)
                
                # Get the processed frame with bounding boxes and use it for display
                processed_frame = results.get("frame", frame)
                
                # Get emotion detection results from audio processing
                if self.emotion_detector:
                    try:
                        # Get the latest audio processing results
                        audio_results = self.emotion_detector.process_audio(
                            None
                        )  # Get latest processed result
                        if audio_results and audio_results.get("emotion") != "unknown":
                            current_emotion = audio_results["emotion"]
                            confidence = audio_results["confidence"]
                            emotion_probs = audio_results.get("emotions", {})
                            
                            # Update current emotion metrics
                            self.metrics["current"]["emotion"] = current_emotion
                            self.metrics["current"]["emotion_confidence"] = confidence

                            # Update emotion history if confidence exceeds threshold
                            if confidence >= self.emotion_detector.threshold:
                                if (
                                    current_emotion
                                    in self.metrics["history"]["emotions"]
                                ):
                                    self.metrics["history"]["emotions"][
                                        current_emotion
                                    ] += 1

                                # Add to emotion log
                                log_entry = {
                                    "timestamp": current_time,
                                    "emotion": current_emotion,
                                    "confidence": confidence,
                                    "message": self._get_emotion_message(
                                        current_emotion, confidence
                                    ),
                                }
                                self._add_to_emotion_log(log_entry)
                            
                            # Emit emotion update via Socket.IO
                                self.socketio.emit(
                                    "emotion_update",
                                    {
                                        "emotion": current_emotion,
                                        "confidence": confidence,
                                        "confidences": emotion_probs,
                                    },
                                )
                    except Exception as e:
                        logger.error(f"Error processing emotion: {e}")
                
                # Update detection metrics and emit update
                if "detections" in results:
                    detection_count = len(results["detections"])
                    # Update metrics
                    self.metrics["current"]["detections"] = detection_count
                    
                    # Update detection types
                    if detection_count > 0:
                        # Increment full-body count for YOLOv8 detections
                        self.metrics["detection_types"]["full_body"] = (
                            self.metrics["detection_types"].get("full_body", 0)
                            + detection_count
                        )
                        
                    # Emit detection update via Socket.IO
                    self.socketio.emit(
                        "detection_update",
                        {
                            "count": detection_count,
                            "fps": self.metrics["current"]["fps"],
                            "detections": results[
                                "detections"
                            ],  # Include actual detection data
                        },
                    )
                
                # Clean up old alerts
                current_time = time.time()
                self.alerts = [
                    a for a in self.alerts if current_time - a["timestamp"] < 60
                ]
                
                # Update alert and notification for crying
                if (
                    self.metrics["current"]["emotion"] == "crying"
                    and self.metrics["current"]["emotion_confidence"]
                    > self.emotion_detector.threshold
                ):
                    # Check if we need to add a new alert
                    if not self.alerts or (
                        current_time - self.alerts[-1]["timestamp"] > 10
                        and self.alerts[-1]["type"] != "crying"
                    ):
                        alert = {
                            "message": "Baby is crying with high confidence!",
                            "type": "crying",
                            "timestamp": current_time,
                        }
                        self.alerts.append(alert)
                        self.socketio.emit("alert", alert)
                
                # Encode and store frame
                with self.frame_lock:
                    # Encode the processed frame with bounding boxes
                    _, buffer = cv2.imencode(".jpg", processed_frame)
                    self.frame_buffer = buffer.tobytes()
                
                # Sleep to control CPU usage
                time.sleep(0.02)  # 50 FPS maximum
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback

                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def _get_emotion_message(self, emotion, confidence):
        """Get a human-readable message for an emotion detection"""
        if emotion == "crying":
            if confidence > 0.85:
                return "Baby is crying loudly! Needs immediate attention."
            elif confidence > 0.7:
                return "Baby is crying. May need attention."
            else:
                return "Baby might be starting to cry."
        elif emotion == "laughing":
            if confidence > 0.8:
                return "Baby is happily laughing!"
            else:
                return "Baby is making happy sounds."
        elif emotion == "babbling":
            return "Baby is babbling or talking."
        elif emotion == "silence":
            return "Baby is quiet."
        else:
            return f"Detected {emotion}."
    
    def _update_system_status(self):
        """Update system status information"""
        while self.running:
            try:
                # Calculate uptime
                uptime = time.time() - self.start_time
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Get emotion state
                current_emotion = "Unknown"
                if self.emotion_detector:
                    current_emotion = self.metrics["current"].get("emotion", "unknown")
                
                # Track peak detections
                current_detections = self.metrics["current"].get("detections", 0)
                if (
                    not hasattr(self, "_peak_detections")
                    or current_detections > self._peak_detections
                ):
                    self._peak_detections = current_detections
                
                # Track total detections (increment by current detection count)
                if not hasattr(self, "_total_detections"):
                    self._total_detections = 0
                self._total_detections += current_detections
                
                # Update metrics with detection data
                self.metrics["peak_detections"] = getattr(self, "_peak_detections", 0)
                self.metrics["total_detections"] = getattr(self, "_total_detections", 0)
                
                # Update system status
                self.system_status.update(
                    {
                        "uptime": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent,
                        "camera_status": (
                            "connected"
                            if self.camera and self.camera.is_opened()
                            else "disconnected"
                        ),
                        "person_detector_status": (
                            "running" if self.person_detector else "stopped"
                        ),
                        "emotion_detector_status": (
                            "running" if self.emotion_detector else "stopped"
                        ),
                        "fps": self.metrics["current"]["fps"],
                        "current_emotion": current_emotion,
                    }
                )
                
                # Emit system status via Socket.IO
                self.socketio.emit("system_info", self.system_status)
                
                # Emit metrics update
                self.socketio.emit(
                    "metrics_update",
                    {
                        "current": {
                            "cpu_usage": self.system_status["cpu_usage"],
                            "memory_usage": self.system_status["memory_usage"],
                            "fps": self.system_status["fps"],
                            "detection_count": current_detections,
                            "emotion": current_emotion,
                            "emotion_confidence": self.metrics["current"].get(
                                "emotion_confidence", 0.0
                            ),
                        },
                        "history": {"emotions": self.metrics["history"]["emotions"]},
                    },
                )
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating system status: {e}")
                time.sleep(1)
    
    def _update_audio_level(self):
        """Update the current audio level in a separate thread."""
        while self.running and not self.stop_event.is_set():
            try:
                if self.emotion_detector:
                    try:
                        # Get current audio level and convert to Python float
                        current_level = float(self.emotion_detector.get_audio_level())
                    except (TypeError, ValueError):
                        # Handle case where get_audio_level returns None or invalid value
                        current_level = -60.0

                    with self.audio_level_lock:
                        self.current_audio_level = current_level

                    # Emit the audio level update with microphone status
                    self.socketio.emit(
                        "audio_level_update",
                        {
                            "level": current_level,
                            "muted": (
                                self.emotion_detector.is_muted
                                if hasattr(self.emotion_detector, "is_muted")
                                else False
                            ),
                            "initialized": (
                                self.emotion_detector.is_initialized
                                if hasattr(self.emotion_detector, "is_initialized")
                                else True
                            ),
                        },
                    )
                else:
                    with self.audio_level_lock:
                        self.current_audio_level = -60.0

                    self.socketio.emit(
                        "audio_level_update",
                        {"level": -60.0, "muted": True, "initialized": False},
                    )

            except Exception as e:
                logger.error(f"Error updating audio level: {str(e)}")
                with self.audio_level_lock:
                    self.current_audio_level = -60.0

            time.sleep(0.1)  # Update every 100ms
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        
    def run(self):
        """Start the web server"""
        self.running = True
        self.processing_thread.start()
        self.status_thread.start()
        self.audio_thread.start()  # Start audio level update thread
        
        try:
            self.socketio.run(
                self.app, host=self.host, port=self.port, debug=self.debug
            )
        except Exception as e:
            logger.error(f"Error running web server: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop the web server"""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()  # Signal threads to stop
        logger.info("Stopping Baby Monitor Web Server")
        
        try:
            # Stop Socket.IO
            if hasattr(self, "socketio"):
                self.socketio.stop()
            
            # Stop camera if it's running
            if self.camera and hasattr(self.camera, "release"):
                self.camera.release()
            
            # Stop emotion detector if it's running
            if self.emotion_detector and hasattr(self.emotion_detector, "stop"):
                self.emotion_detector.stop()
            
            # Stop person detector if it's running
            if self.person_detector and hasattr(self.person_detector, "stop"):
                self.person_detector.stop()
            
            # Wait for threads to finish
            if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            if hasattr(self, "status_thread") and self.status_thread.is_alive():
                self.status_thread.join(timeout=1.0)
            
            if hasattr(self, "audio_thread") and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
        finally:
            logger.info("Web server stopped")
            os._exit(0)  # Force exit if normal shutdown fails 
