import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from datetime import datetime
import os

# Create output directory
output_dir = "/Users/soumyashekhar/Desktop/FILES/Projects/SKYGRID/saved_frames"
os.makedirs(output_dir, exist_ok=True)

def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture('/Users/soumyashekhar/Desktop/3.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps
frame_count = 0

MIN_CONFIDENCE = 0.4
MIN_PERSON_SIZE = 2
MAX_PERSON_SIZE = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_interval != 0:
        continue
        
    frame = preprocess_frame(frame)
    height, width = frame.shape[:2]
    scale = min(1920 / width, 1080 / height)
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=MIN_CONFIDENCE)[0]
    
    human_count = 0
    detections = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:
            box = box.tolist()
            person_height = box[3] - box[1]
            person_width = box[2] - box[0]
            
            if MIN_PERSON_SIZE <= person_height <= MAX_PERSON_SIZE and \
               MIN_PERSON_SIZE <= person_width <= MAX_PERSON_SIZE:
                x1, y1, x2, y2 = map(int, box)
                confidence = score.item()
                detections.append({
                    'confidence': confidence,
                    'position': (x1, y1, x2, y2),
                    'size': (person_width, person_height)
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                human_count += 1

    # Add count to top right corner
    text = f'Count: {human_count}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = width - text_size[0] - 10
    cv2.putText(frame, text, (text_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Frame {frame_count} Analysis ({timestamp}) ===")
    print(f"Total Humans Detected: {human_count}")
    
    if human_count > 0:
        print("\nDetailed Detection Information:")
        for i, det in enumerate(detections, 1):
            print(f"Detection {i}:")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  Position (x1,y1,x2,y2): {det['position']}")
            print(f"  Size (w,h): ({det['size'][0]:.1f}, {det['size'][1]:.1f})")
        
        # Save frame with detections
        save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{frame_count}_humans_{human_count}_{save_timestamp}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        print(f"\nFrame saved as: {filename}")
    
    print("=" * 50)
    
    cv2.putText(frame, f'Humans: {human_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('High-Altitude Human Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()