import os
import requests

def download_file(url, filename):
    """Download a file from a URL to a specific filename."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models/mobilenet_ssd", exist_ok=True)
    
    # Download model files
    files = {
        "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    for filename, url in files.items():
        filepath = os.path.join("models/mobilenet_ssd", filename)
        download_file(url, filepath)

if __name__ == "__main__":
    main() 