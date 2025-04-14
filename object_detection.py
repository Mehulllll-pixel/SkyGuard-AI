import torch
import cv2
from pathlib import Path

# Load YOLOv5 model (using 'yolov5s' for fast inference)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image path (replace with the path to your image or video)
image_path = "test_image.jpg"



# Load image using OpenCV
img = cv2.imread(image_path)

# Perform object detection
results = model(img)

# Display the results (with bounding boxes and labels)
results.show()

# Save the results to a folder
results.save(Path('results'))  # Save to 'results' folder

# Print detected labels and confidence scores
print(f"Detected labels: {results.names}")
print(f"Bounding boxes: {results.xywh[0]}")
print(f"Confidence scores: {results.confidence}")
