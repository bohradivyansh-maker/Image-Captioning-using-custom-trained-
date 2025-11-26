"""
Download YOLOv3-tiny model files for object detection.
"""
import os
import urllib.request

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# YOLOv3-tiny files (smaller and faster than full YOLOv3)
files_to_download = {
    'yolov3-tiny.weights': 'https://pjreddie.com/media/files/yolov3-tiny.weights',
    'yolov3-tiny.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
    'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
}

print("Downloading YOLO model files...")

for filename, url in files_to_download.items():
    filepath = os.path.join('models', filename)
    
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists, skipping...")
        continue
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ {filename} downloaded successfully!")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print("\nYOLO model setup complete!")
print("You can now run the app with: streamlit run app.py")
