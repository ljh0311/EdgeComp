import torch
from ultralytics import YOLO
import platform
import time
import os


class PersonDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        """Initialize PersonDetector with YOLO model.
        
        Args:
            model_path (str): Path to YOLO model file. If relative, will be resolved from project root.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set quantization engine based on platform
        if platform.system() == "Windows":
            torch.backends.quantized.engine = "onednn"
        elif platform.system() == "Linux":
            # Check if running on Raspberry Pi
            try:
                with open("/proc/cpuinfo", "r") as f:
                    if "Raspberry Pi" in f.read():
                        torch.backends.quantized.engine = "qnnpack"
            except:
                torch.backends.quantized.engine = "onednn"

        # Ensure model path is absolute
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load and quantize model
        self.model = YOLO(model_path)
        if self.device == "cpu":
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


if __name__ == "__main__":
    import cv2
    import argparse

    def create_flask_app():
        """Create Flask app for person detection web server"""
        from flask import Flask, Response, render_template
        import cv2

        app = Flask(__name__)
        detector = None
        camera = None

        def get_camera():
            """Initialize camera only when needed"""
            nonlocal camera
            if camera is None:
                camera = cv2.VideoCapture(0)
            return camera

        def get_detector():
            """Initialize detector only when needed"""
            nonlocal detector
            if detector is None:
                detector = PersonDetector()
            return detector

        def generate_frames():
            """Generate video frames with person detection"""
            while True:
                camera = get_camera()
                success, frame = camera.read()
                if not success:
                    break

                # Run detection
                detector = get_detector()
                results, fps = detector.detect(frame)

                # Draw results
                annotated_frame = results.plot()

                # Add FPS text
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Convert to bytes
                ret, buffer = cv2.imencode(".jpg", annotated_frame)
                frame = buffer.tobytes()

                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

        @app.route("/")
        def index():
            """Render main page"""
            return render_template("index.html")

        @app.route("/video_feed")
        def video_feed():
            """Video streaming route"""
            return Response(
                generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @app.teardown_appcontext
        def cleanup(exception):
            """Clean up resources"""
            if camera is not None:
                camera.release()

        return app

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test person detection")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source video. Can be video file path or camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model (default: yolov8n.pt)",
    )
    args = parser.parse_args()

    # Initialize detector
    detector = PersonDetector(args.model)
    print(f"Running inference on {detector.device}")

    # Open video source
    try:
        source = int(args.source)  # Try camera index first
    except ValueError:
        source = args.source  # Otherwise treat as file path
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results, fps = detector.detect(frame)

        # Draw results
        annotated_frame = results.plot()

        # Add FPS counter
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display
        cv2.imshow("Person Detection", annotated_frame)

        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
