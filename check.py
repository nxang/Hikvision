import cv2
import matplotlib.pyplot as plt

def plot_annotations(image_path, label_path):
    # Load image
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Read YOLO coordinates
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # Handle standard bounding boxes (5 elements)
            if len(parts) == 5:
                cls, x, y, bbox_w, bbox_h = map(float, parts)
                # Convert normalized back to absolute pixels
                x1 = int((x - bbox_w / 2) * w)
                y1 = int((y - bbox_h / 2) * h)
                x2 = int((x + bbox_w / 2) * w)
                y2 = int((y + bbox_h / 2) * h)
                # Draw tight green box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage:
plot_annotations("Fish.v1i.yolo26/test/images/Preset_2_20260526_150753_jpg.rf.f1b1fbe11f5d74ecfb0c01f122d8b1fd.jpg", "Fish.v1i.yolo26/test/labels/Preset_2_20260526_150753_jpg.rf.f1b1fbe11f5d74ecfb0c01f122d8b1fd.txt")