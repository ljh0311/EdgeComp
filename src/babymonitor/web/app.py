"""
Web Interface Module
-------------------
Handles the web interface for the Baby Monitor System.
"""

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    send_from_directory,
)
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Dict, Any
import os
import base64
import json
from pathlib import Path

# Import API route definitions
from .routes import (
    INDEX_PAGE,
    METRICS_PAGE,
    REPAIR_TOOLS,
    VIDEO_FEED,
    SYSTEM_STATUS,
    SYSTEM_INFO,
    SYSTEM_RESTART,
    SYSTEM_STOP,
    CAMERA_LIST,
    CAMERA_ADD,
    CAMERA_REMOVE,
    CAMERA_ACTIVATE,
    AUDIO_MICROPHONES,
    AUDIO_SET_MICROPHONE,
    AUDIO_LEVEL,
    AUDIO_INFO,
    EMOTION_MODELS,
    EMOTION_MODEL_INFO,
    EMOTION_SWITCH_MODEL,
    EMOTION_TEST_AUDIO,
    EMOTION_RESTART_AUDIO,
)

# Import detectors
from ..detectors.person_detector import PersonDetector
from ..detectors.emotion_detector import EmotionDetector
from ..config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BabyMonitorWeb:
    """Web interface for the Baby Monitor System."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, mode: str = "normal"):
        """Initialize the web interface.

        Args:
            host: Host address to bind to
            port: Port to listen on
            mode: Operation mode ('normal' or 'dev')
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.dev_mode = mode == "dev"

        # Set up Flask with correct template directory
        template_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "templates")
        )
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
        self.app = Flask(
            __name__, template_folder=template_dir, static_folder=static_dir
        )

        self.socketio = SocketIO(self.app, cors_allowed_origins="*", json=json)

        # Initialize detectors with proper error handling
        try:
            self.person_detector = PersonDetector(
                threshold=Config.PERSON_DETECTION.get("threshold", 0.7)
            )
            self.emotion_detector = EmotionDetector(
                threshold=Config.EMOTION_DETECTION.get("threshold", 0.7)
            )

            # Verify model initialization
            if not hasattr(self.person_detector, "is_model_loaded"):
                self.person_detector.is_model_loaded = False
            if not hasattr(self.emotion_detector, "is_model_loaded"):
                self.emotion_detector.is_model_loaded = False

            logger.info("Detectors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detectors: {str(e)}")
            self.person_detector = None
            self.emotion_detector = None

        # Initialize camera manager
        try:
            from ..camera.camera_manager import CameraManager

            self.camera_manager = CameraManager()
            logger.info("Camera manager initialized successfully")
        except ImportError:
            logger.error("CameraManager module not found, trying alternate import path")
            try:
                from ..utils.camera import CameraManager

                self.camera_manager = CameraManager()
                logger.info("Camera manager initialized successfully from utils.camera")
            except ImportError:
                logger.error("Failed to import CameraManager from any module")
                self.camera_manager = None
        except Exception as e:
            logger.error(f"Error initializing camera manager: {str(e)}")
            self.camera_manager = None

        # Thread safety
        self.frame_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.metrics_lock = threading.Lock()

        # Queues for data
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)

        # Thread management
        self.is_running = False
        self.video_thread = None
        self.audio_thread = None
        self.stop_event = threading.Event()

        # Video capture
        self.video_capture = None
        self.camera_id = 0
        self.active_camera = None

        # System metrics
        self.start_time = time.time()
        self.detection_history = []
        self.emotion_history = []

        # Set up routes
        self.setup_routes()

        # Set up socket events
        self.setup_socketio()

        # Start processing threads
        self.start_processing()

    def setup_routes(self):
        """Set up Flask routes."""

        @self.app.route(INDEX_PAGE)
        def index():
            return render_template("index.html", mode=self.mode, dev_mode=self.dev_mode)

        @self.app.route(METRICS_PAGE)
        def metrics():
            return render_template(
                "metrics.html", mode=self.mode, dev_mode=self.dev_mode
            )

        @self.app.route(REPAIR_TOOLS)
        def repair_tools():
            return render_template(
                "repair_tools.html", mode=self.mode, dev_mode=self.dev_mode
            )

        @self.app.route(VIDEO_FEED)
        def video_feed():
            return Response(
                self.generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route(AUDIO_INFO)
        def get_microphone_info():
            """Get current microphone information."""
            try:
                if not hasattr(self, 'emotion_detector') or self.emotion_detector is None:
                    return jsonify({
                        'id': '0',
                        'name': 'Not available',
                        'status': 'error',
                        'error': 'Emotion detector not initialized'
                    }), 200  # Return 200 with error info instead of 500

                mic_id = getattr(self.emotion_detector, 'current_microphone_id', '0')
                mic_name = "Default Microphone"
                try:
                    import sounddevice as sd
                    device_info = sd.query_devices(int(mic_id)) if mic_id else None
                    if device_info:
                        mic_name = device_info.get('name', 'Default Microphone')
                except Exception as e:
                    logger.warning(f"Could not get microphone name: {e}")
                    
                return jsonify({
                    'id': str(mic_id),
                    'name': mic_name,
                    'status': 'active' if not getattr(self.emotion_detector, 'is_muted', True) else 'muted'
                }), 200

            except Exception as e:
                logger.error(f"Error getting microphone info: {e}")
                return jsonify({
                    'id': '0',
                    'name': 'Not available',
                    'status': 'error',
                    'error': str(e)
                }), 200  # Return 200 with error info

        @self.app.route(CAMERA_LIST)
        def list_cameras():
            """Get list of available cameras."""
            try:
                if not hasattr(self, 'camera_manager') or self.camera_manager is None:
                    return jsonify({
                        'cameras': [{
                            'id': '0',
                            'name': 'Default Camera',
                            'active': True
                        }]
                    }), 200

                cameras = self.camera_manager.get_cameras()
                return jsonify({
                    'cameras': [{
                        'id': str(cam.get('id', i)),
                        'name': cam.get('name', f'Camera {i+1}'),
                        'active': cam.get('active', False) or (
                            hasattr(self, 'active_camera') and 
                            self.active_camera == str(cam.get('id', i))
                        )
                    } for i, cam in enumerate(cameras)]
                }), 200

            except Exception as e:
                logger.error(f"Error listing cameras: {e}")
                return jsonify({
                    'cameras': [],
                    'error': str(e)
                }), 200  # Return 200 with empty list and error

        @self.app.route(SYSTEM_STATUS)
        def get_system_status_api():
            try:
                status = self.get_system_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error checking system status: {str(e)}")
                return (
                    jsonify(
                        {
                            "camera": {"status": "error", "message": str(e)},
                            "audio": {"status": "error", "message": str(e)},
                            "detection": {"status": "error", "message": str(e)},
                            "emotion": {"status": "error", "message": str(e)},
                        }
                    ),
                    500,
                )

        @self.app.route(AUDIO_MICROPHONES)
        def get_microphones():
            try:
                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return jsonify(
                        {
                            "microphones": [],
                            "current_microphone": None,
                            "error": "Audio system not initialized",
                        }
                    )
                microphones = self.emotion_detector.get_available_devices()
                current_mic = getattr(
                    self.emotion_detector, "current_microphone_id", None
                )
                return jsonify(
                    {"microphones": microphones, "current_microphone": current_mic}
                )
            except Exception as e:
                logger.error(f"Error getting microphones: {str(e)}")
                return jsonify(
                    {"microphones": [], "current_microphone": None, "error": str(e)}
                )

        @self.app.route(AUDIO_SET_MICROPHONE, methods=["POST"])
        def set_microphone():
            try:
                data = request.get_json()
                if not data or "microphone_id" not in data:
                    return jsonify({"error": "Missing microphone_id"}), 400

                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return jsonify({"error": "Audio system not initialized"}), 500

                success = self.emotion_detector.set_microphone(data["microphone_id"])
                if success:
                    return jsonify(
                        {
                            "status": "success",
                            "message": "Microphone set successfully",
                            "microphone_id": data["microphone_id"],
                        }
                    )
                return jsonify({"error": "Failed to set microphone"}), 500
            except Exception as e:
                logger.error(f"Error setting microphone: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route(AUDIO_LEVEL)
        def get_audio_level():
            try:
                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return jsonify({"error": "Audio system not initialized"}), 500

                level = self.emotion_detector.get_audio_level()
                return jsonify({"level": level})
            except Exception as e:
                logger.error(f"Error getting audio level: {str(e)}")
                return jsonify({"error": str(e)}), 500

        # Emotion detection endpoints
        @self.app.route(EMOTION_MODELS, methods=["GET"])
        def get_emotion_models():
            """Get list of available emotion models."""
            try:
                logger.info("Fetching available emotion models...")

                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    logger.error("Emotion detector not initialized")
                    return (
                        jsonify(
                            {
                                "error": "Emotion detector not initialized",
                                "models": [],
                                "current_model": {
                                    "id": "unknown",
                                    "name": "Unknown",
                                    "type": "Unknown",
                                    "emotions": [],
                                },
                            }
                        ),
                        500,
                    )

                models_data = self.emotion_detector.get_available_models()
                logger.info(f"Found {len(models_data['models'])} models")
                return jsonify(models_data)
            except Exception as e:
                logger.error(f"Error fetching emotion models: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                return (
                    jsonify(
                        {
                            "error": str(e),
                            "models": [],
                            "current_model": {
                                "id": "unknown",
                                "name": "Unknown",
                                "type": "Unknown",
                                "emotions": [],
                            },
                        }
                    ),
                    500,
                )

        @self.app.route(f"{EMOTION_MODEL_INFO}/<model_id>", methods=["GET"])
        def get_model_info(model_id):
            """Get information about a specific emotion model."""
            try:
                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return jsonify({"error": "Emotion detector not initialized"}), 500

                models_data = self.emotion_detector.get_available_models()
                for model in models_data["models"]:
                    if model["id"] == model_id:
                        return jsonify({"model_info": model})

                return jsonify({"error": f"Model with ID {model_id} not found"}), 404
            except Exception as e:
                logger.error(f"Error fetching model info: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route(EMOTION_SWITCH_MODEL, methods=["POST"])
        def switch_emotion_model():
            """Switch to a different emotion recognition model."""
            try:
                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return jsonify({"error": "Emotion detector not initialized"}), 500

                data = request.get_json()
                if not data:
                    return jsonify({"error": "No data provided"}), 400

                model_id = data.get("model_id")
                if not model_id:
                    return jsonify({"error": "Model ID is required"}), 400

                try:
                    result = self.emotion_detector.switch_model(model_id)
                    # Convert the result to be compatible with the frontend
                    response = {
                        "status": "success",
                        "message": f"Switched to model: {result['model_info']['name']}",
                        "model_info": result["model_info"],
                    }
                    # Emit an event to all clients
                    self.socketio.emit("emotion_model_changed", result["model_info"])
                    return jsonify(response)
                except ValueError as e:
                    logger.error(f"Invalid model ID: {str(e)}")
                    return jsonify({"status": "error", "message": str(e)}), 400
                except Exception as e:
                    logger.error(f"Error switching model: {str(e)}")
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Failed to switch model: {str(e)}",
                            }
                        ),
                        500,
                    )
            except Exception as e:
                logger.error(f"Unexpected error switching model: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"status": "error", "message": str(e)}), 500

        # Audio test and restart endpoints (new emotion-specific paths)
        @self.app.route(EMOTION_TEST_AUDIO, methods=["POST"])
        def emotion_test_audio():
            """Test audio system by recording 5 seconds of live audio."""
            try:
                if (
                    not hasattr(self, "emotion_detector")
                    or self.emotion_detector is None
                ):
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Emotion detector not initialized",
                            }
                        ),
                        500,
                    )

                # Record 5 seconds of audio
                audio_data = self.emotion_detector.record_audio(duration=5)
                
                # Convert audio data to base64 for transmission
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # Get emotion detection results
                emotion_result = self.emotion_detector.analyze_audio(audio_data)

                response = {
                    "status": "success",
                    "message": "Audio test completed",
                    "audio": audio_b64,
                    "emotion": emotion_result["emotion"],
                    "confidence": emotion_result["confidence"],
                    "confidences": emotion_result["confidences"]
                }

                # Emit results to all connected clients
                self.socketio.emit("emotion_update", emotion_result)
                self.socketio.emit("audio_recorded", {"audio": audio_b64})

                return jsonify(response)

            except Exception as e:
                logger.error(f"Error testing audio: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route(EMOTION_RESTART_AUDIO, methods=['POST'])
        def emotion_restart_audio():
            """Restart audio system."""
            try:
                if not hasattr(self, 'emotion_detector') or self.emotion_detector is None:
                    return jsonify({'status': 'error', 'message': 'Audio system not initialized'}), 500
                    
                self.emotion_detector.restart_audio()
                return jsonify({
                    'status': 'success',
                    'message': 'Audio system restarted successfully'
                })
            except Exception as e:
                logger.error(f"Error restarting audio: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        # Camera management endpoints
        @self.app.route(CAMERA_ADD, methods=["POST"])
        def add_camera():
            """Add a new camera."""
            try:
                if not hasattr(self, "camera_manager") or self.camera_manager is None:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Camera manager not initialized",
                            }
                        ),
                        500,
                    )

                data = request.get_json()
                if not data:
                    return (
                        jsonify({"status": "error", "message": "No data provided"}),
                        400,
                    )

                name = data.get("name")
                device = data.get("device")

                if not name:
                    return (
                        jsonify(
                            {"status": "error", "message": "Camera name is required"}
                        ),
                        400,
                    )

                if not device:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Device ID or URL is required",
                            }
                        ),
                        400,
                    )

                # Convert to int if it's a number
                try:
                    if device.isdigit():
                        device = int(device)
                except:
                    # It's a string URL
                    pass

                success = self.camera_manager.add_camera(name, device)

                if success:
                    # If this is the first camera, make it active
                    if not self.active_camera:
                        self.active_camera = name
                        self.switch_to_camera(name)

                    # Notify all clients
                    self.socketio.emit(
                        "camera_added",
                        {
                            "name": name,
                            "device": str(device),
                            "active": self.active_camera == name,
                        },
                    )

                    return jsonify(
                        {
                            "status": "success",
                            "message": f'Camera "{name}" added successfully',
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f'Failed to add camera "{name}"',
                            }
                        ),
                        500,
                    )

            except Exception as e:
                logger.error(f"Error adding camera: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route(CAMERA_REMOVE, methods=["POST"])
        def remove_camera():
            """Remove a camera."""
            try:
                if not hasattr(self, "camera_manager") or self.camera_manager is None:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Camera manager not initialized",
                            }
                        ),
                        500,
                    )

                data = request.get_json()
                if not data:
                    return (
                        jsonify({"status": "error", "message": "No data provided"}),
                        400,
                    )

                camera_id = data.get("camera_id")

                if not camera_id:
                    return (
                        jsonify(
                            {"status": "error", "message": "Camera ID is required"}
                        ),
                        400,
                    )

                # If this is the active camera, switch to another camera first
                if self.active_camera == camera_id:
                    self.active_camera = None
                    self.stop_camera()
                    # Find another camera to switch to
                    camera_list = self.camera_manager.get_camera_list()
                    if camera_list and len(camera_list) > 1:
                        for camera in camera_list:
                            if camera["name"] != camera_id:
                                self.active_camera = camera["name"]
                                self.switch_to_camera(camera["name"])
                                break

                success = self.camera_manager.remove_camera(camera_id)

                if success:
                    # Notify all clients
                    self.socketio.emit("camera_removed", {"id": camera_id})

                    return jsonify(
                        {"status": "success", "message": f"Camera removed successfully"}
                    )
                else:
                    return (
                        jsonify(
                            {"status": "error", "message": f"Failed to remove camera"}
                        ),
                        500,
                    )

            except Exception as e:
                logger.error(f"Error removing camera: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route(CAMERA_ACTIVATE, methods=["POST"])
        def activate_camera():
            """Activate a specific camera."""
            try:
                if not hasattr(self, "camera_manager") or self.camera_manager is None:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Camera manager not initialized",
                            }
                        ),
                        500,
                    )

                data = request.get_json()
                if not data:
                    return (
                        jsonify({"status": "error", "message": "No data provided"}),
                        400,
                    )

                camera_id = data.get("camera_id")

                if not camera_id:
                    return (
                        jsonify(
                            {"status": "error", "message": "Camera ID is required"}
                        ),
                        400,
                    )

                # Check if camera exists
                camera = self.camera_manager.get_camera(camera_id)
                if not camera:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Camera not found: {camera_id}",
                            }
                        ),
                        404,
                    )

                # Switch to this camera
                success = self.switch_to_camera(camera_id)

                if success:
                    self.active_camera = camera_id

                    # Notify all clients
                    self.socketio.emit("camera_activated", {"id": camera_id})

                    return jsonify(
                        {
                            "status": "success",
                            "message": f"Camera activated successfully",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {"status": "error", "message": f"Failed to activate camera"}
                        ),
                        500,
                    )

            except Exception as e:
                logger.error(f"Error activating camera: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route(VIDEO_FEED)
        def camera_feed(camera_id):
            """Video streaming route for specific camera."""
            if not hasattr(self, "camera_manager") or self.camera_manager is None:
                return jsonify({"error": "Camera manager not initialized"}), 500

            camera = self.camera_manager.get_camera(camera_id)
            if not camera:
                return jsonify({"error": f"Camera not found: {camera_id}"}), 404

            def generate():
                """Video streaming generator function."""
                while True:
                    try:
                        if not camera.cap.isOpened():
                            # Try to reinitialize
                            if not camera.initialize():
                                logger.error(
                                    f"Failed to reinitialize camera: {camera_id}"
                                )
                                break

                        success, frame = camera.cap.read()
                        if not success:
                            logger.error(
                                f"Failed to read frame from camera: {camera_id}"
                            )
                            break

                        # Encode frame as JPEG
                        ret, jpeg = cv2.imencode(".jpg", frame)
                        if not ret:
                            logger.error(
                                f"Failed to encode frame from camera: {camera_id}"
                            )
                            break

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg.tobytes()
                            + b"\r\n"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating frame for camera {camera_id}: {str(e)}"
                        )
                        break

            return Response(
                generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        # System control endpoints
        @self.app.route(SYSTEM_RESTART, methods=["POST"])
        def restart_system():
            """Restart the system."""
            try:
                # Stop current processing
                if hasattr(self, "stop_processing"):
                    self.stop_processing()
                else:
                    logger.warning("stop_processing method not available")

                # Reinitialize detectors
                try:
                    if hasattr(self, "person_detector"):
                        self.person_detector = PersonDetector(
                            threshold=Config.PERSON_DETECTION.get("threshold", 0.7)
                        )

                    if hasattr(self, "emotion_detector"):
                        self.emotion_detector = EmotionDetector(
                            threshold=Config.EMOTION_DETECTION.get("threshold", 0.7)
                        )

                    # Verify model initialization
                    if hasattr(self, "person_detector") and not hasattr(
                        self.person_detector, "is_model_loaded"
                    ):
                        self.person_detector.is_model_loaded = False

                    if hasattr(self, "emotion_detector") and not hasattr(
                        self.emotion_detector, "is_model_loaded"
                    ):
                        self.emotion_detector.is_model_loaded = False
                except Exception as e:
                    logger.error(f"Error reinitializing detectors: {str(e)}")
                    return (
                        jsonify(
                            {"error": "Failed to reinitialize detectors: " + str(e)}
                        ),
                        500,
                    )

                # Restart processing
                if hasattr(self, "start_processing"):
                    self.start_processing()
                else:
                    logger.warning("start_processing method not available")

                return jsonify(
                    {"status": "success", "message": "System restarted successfully"}
                )
            except Exception as e:
                logger.error(f"Error restarting system: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route(SYSTEM_STOP, methods=["POST"])
        def stop_system():
            """Stop the system."""
            try:
                if hasattr(self, "stop_processing"):
                    self.stop_processing()
                else:
                    logger.warning("stop_processing method not available")

                return jsonify(
                    {"status": "success", "message": "System stopped successfully"}
                )
            except Exception as e:
                logger.error(f"Error stopping system: {str(e)}")
                return jsonify({"error": str(e)}), 500

    def setup_socketio(self):
        """Set up Socket.IO event handlers."""

        @self.socketio.on("connect")
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            # Send initial system status
            emit("system_status", self.get_system_status())

        @self.socketio.on("disconnect")
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")

        @self.socketio.on("request_camera_list")
        def handle_camera_list_request():
            """Handle request for camera list."""
            try:
                cameras = []
                # Add the current camera if it exists
                if (
                    hasattr(self, "video_capture")
                    and self.video_capture
                    and self.video_capture.isOpened()
                ):
                    cameras.append(
                        {
                            "name": f"Camera {self.camera_id}",
                            "source": "USB Camera",
                            "status": "Active",
                        }
                    )

                # Here you would typically add logic to fetch other registered cameras
                # For demonstration, add a fake IP camera if none exists
                if not cameras:
                    cameras.append(
                        {
                            "name": "Demo Camera",
                            "source": "USB Camera 0",
                            "status": "Inactive",
                        }
                    )

                logger.info(f"Sending camera list: {len(cameras)} cameras")
                emit("camera_list", {"cameras": cameras})
            except Exception as e:
                logger.error(f"Error getting camera list: {str(e)}")
                emit("error", {"message": f"Failed to get camera list: {str(e)}"})

        @self.socketio.on("toggle_camera")
        def handle_toggle_camera():
            """Toggle the current camera on/off."""
            try:
                camera_enabled = False
                if (
                    hasattr(self, "video_capture")
                    and self.video_capture
                    and self.video_capture.isOpened()
                ):
                    self.stop_camera()
                    camera_enabled = False
                    logger.info("Camera stopped via socket.io")
                else:
                    self.start_camera()
                    camera_enabled = (
                        hasattr(self, "video_capture")
                        and self.video_capture
                        and self.video_capture.isOpened()
                    )
                    logger.info("Camera started via socket.io")

                emit("camera_status", {"enabled": camera_enabled})

                # Also update system status
                emit("system_status", self.get_system_status())

                # Update camera list
                handle_camera_list_request()
            except Exception as e:
                logger.error(f"Error toggling camera: {str(e)}")
                emit("error", {"message": f"Failed to toggle camera: {str(e)}"})

    def start_camera(self):
        """Start the camera."""
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(self.camera_id)
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            logger.info("Camera started")
            return True
        return False

    def stop_camera(self):
        """Stop the camera."""
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
            logger.info("Camera stopped")
            return True
        return False

    def start_processing(self):
        """Start processing threads."""
        self.is_running = True
        self.stop_event.clear()

        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()

        logger.info("Processing threads started")

    def stop_processing(self):
        """Stop processing threads."""
        self.is_running = False
        self.stop_event.set()

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)

        self.stop_camera()
        logger.info("Processing threads stopped")

    def process_video(self):
        """Process video frames."""
        logger.info("Video processing thread started")

        # Start camera if not already started
        if not self.start_camera():
            logger.error("Failed to start camera")
            return

        while self.is_running and not self.stop_event.is_set():
            try:
                if self.video_capture is not None and self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    if not ret:
                        logger.error("Failed to read frame from camera")
                        time.sleep(0.1)
                        continue

                    # Process frame with person detector
                    result = self.person_detector.process_frame(frame)
                    processed_frame = result["frame"]

                    # Put frame in queue for streaming
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame)

                    # Prepare detection data for Socket.IO
                    detections = result.get("detections", [])

                    # Ensure all values are JSON serializable
                    detection_data = {
                        "count": len(detections),
                        "fps": float(result.get("fps", 0)),
                        "detections": [
                            {
                                "bbox": [
                                    float(x) for x in det.get("bbox", [0, 0, 0, 0])
                                ],
                                "confidence": float(det.get("confidence", 0)),
                                "class": str(det.get("class", "unknown")),
                            }
                            for det in detections
                        ],
                    }

                    # Add to detection history
                    self.detection_history.append(
                        {
                            "timestamp": time.time(),
                            "count": len(detections),
                            "types": {
                                det_type: sum(
                                    1 for d in detections if d.get("class") == det_type
                                )
                                for det_type in set(
                                    d.get("class", "unknown") for d in detections
                                )
                            },
                        }
                    )

                    # Limit history size
                    if len(self.detection_history) > 100:
                        self.detection_history.pop(0)

                    # Emit detection data via Socket.IO
                    self.socketio.emit("detection_update", detection_data)

                    # Emit metrics
                    metrics = {
                        "fps": float(self.person_detector.fps),
                        "detection_count": len(detections),
                        "detection_types": {
                            str(det_type): sum(
                                1 for d in detections if d.get("class") == det_type
                            )
                            for det_type in set(
                                d.get("class", "unknown") for d in detections
                            )
                        },
                    }
                    self.socketio.emit("metrics_update", {"current": metrics})

                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in video processing: {e}")
                time.sleep(0.1)

    def process_audio(self):
        """Process audio chunks."""
        logger.info("Audio processing thread started")

        # Set up audio capture
        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if not self.audio_queue.full():
                    self.audio_queue.put(indata.copy())

            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=self.emotion_detector.SAMPLE_RATE,
                blocksize=self.emotion_detector.CHUNK_SIZE,
                callback=audio_callback,
            ):
                while self.is_running and not self.stop_event.is_set():
                    try:
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get(timeout=0.1)

                            # Process audio with emotion detector
                            result = self.emotion_detector.process_audio(audio_data)

                            # Emit emotion data via Socket.IO
                            if result["emotion"] not in [
                                "buffering",
                                "unknown",
                                "error",
                            ]:
                                # Ensure all values are JSON serializable
                                emotion_data = {
                                    "emotion": str(result["emotion"]),
                                    "confidence": float(result["confidence"]),
                                    "emotions": {
                                        str(k): float(v)
                                        for k, v in result["emotions"].items()
                                    },
                                }

                                # Add to emotion history
                                self.emotion_history.append(
                                    {
                                        "timestamp": time.time(),
                                        "emotion": result["emotion"],
                                        "confidence": float(result["confidence"]),
                                    }
                                )

                                # Limit history size
                                if len(self.emotion_history) > 100:
                                    self.emotion_history.pop(0)

                                self.socketio.emit("emotion_update", emotion_data)

                                # Emit alert for crying
                                if (
                                    result["emotion"] == "crying"
                                    and result["confidence"]
                                    > self.emotion_detector.threshold
                                ):
                                    self.socketio.emit(
                                        "alert",
                                        {
                                            "type": "crying",
                                            "message": "Baby is crying!",
                                            "timestamp": time.time(),
                                        },
                                    )
                        else:
                            time.sleep(0.01)

                    except queue.Empty:
                        time.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error in audio processing: {e}")
                        time.sleep(0.1)

        except ImportError:
            logger.error("sounddevice not installed, audio processing disabled")
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")

    def generate_frames(self):
        """Generate frames for video streaming."""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            except queue.Empty:
                # Return an empty frame if queue is empty
                empty_frame = np.zeros(
                    (Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8
                )
                _, buffer = cv2.imencode(".jpg", empty_frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            except Exception as e:
                logger.error(f"Error generating frames: {e}")
                time.sleep(0.1)

    def start(self):
        """Start the web interface."""
        try:
            logger.info(
                f"Baby Monitor web system started on http://{self.host}:{self.port}"
            )
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the web interface."""
        try:
            self.stop_processing()

            # Clean up detectors
            if hasattr(self, "person_detector"):
                self.person_detector.cleanup()

            if hasattr(self, "emotion_detector"):
                self.emotion_detector.cleanup()

            logger.info("Web interface stopped")
        except Exception as e:
            logger.error(f"Error stopping web interface: {e}")
            raise

    @property
    def is_system_ready(self):
        """Check if the system is ready."""
        return (
            self.person_detector is not None
            and self.emotion_detector is not None
            and hasattr(self.emotion_detector, "model_info")
        )

    def get_system_status(self):
        """Get the current system status."""
        if not self.is_system_ready:
            return {
                "camera": {"status": "error", "message": "System not initialized"},
                "audio": {"status": "error", "message": "System not initialized"},
                "detection": {"status": "error", "message": "System not initialized"},
                "emotion": {"status": "error", "message": "System not initialized"},
            }

        return {
            "camera": {
                "status": (
                    "ok"
                    if self.video_capture and self.video_capture.isOpened()
                    else "error"
                ),
                "message": (
                    "Camera is active"
                    if self.video_capture and self.video_capture.isOpened()
                    else "Camera is not available"
                ),
            },
            "audio": {"status": "ok", "message": "Audio system is active"},
            "detection": {
                "status": (
                    "ok"
                    if getattr(self.person_detector, "is_model_loaded", False)
                    else "error"
                ),
                "message": (
                    "Person detection is active"
                    if getattr(self.person_detector, "is_model_loaded", False)
                    else "Person detection model not loaded"
                ),
            },
            "emotion": {
                "status": (
                    "ok"
                    if getattr(self.emotion_detector, "is_model_loaded", False)
                    else "error"
                ),
                "message": (
                    "Emotion detection is active"
                    if getattr(self.emotion_detector, "is_model_loaded", False)
                    else "Emotion detection model not loaded"
                ),
            },
        }

    def switch_to_camera(self, camera_name):
        """Switch video feed to a specific camera.

        Args:
            camera_name: Name of the camera to switch to

        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Switching to camera: {camera_name}")

            # Get camera from manager
            camera = self.camera_manager.get_camera(camera_name)
            if not camera:
                logger.error(f"Camera not found: {camera_name}")
                return False

            # Stop current camera if running
            self.stop_camera()

            # Set this camera as the active source
            self.video_capture = camera.cap
            self.active_camera = camera_name

            if not self.video_capture.isOpened():
                if not camera.initialize():
                    logger.error(f"Failed to open camera: {camera_name}")
                    return False
                self.video_capture = camera.cap

            logger.info(f"Successfully switched to camera: {camera_name}")
            return True

        except Exception as e:
            logger.error(f"Error switching camera: {str(e)}")
            return False


if __name__ == "__main__":
    web_app = BabyMonitorWeb()
    web_app.start()
