import os

from ultralytics import YOLO
import cv2
import glob

# 1. Load your trained model weights
model_path = YOLO(r"runs\detect\aquaculture_pellet\pellet_run_s\weights\best.pt")

# 2. Run prediction on your image
folder_path = "Captures_2026-05-29"
output_dir = r"C:\Users\yieku\Desktop\Hikvision\runs\detect\pellets"
os.makedirs(output_dir, exist_ok=True)

# 2. Load model and find images
model = YOLO(model_path)
images_list = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))

# 3. Define the maximum allowable size for a single pellet (in pixels)
# If a box is wider or taller than this number, the script deletes it.
MAX_PELLET_SIZE = 10000  # Adjust this based on your image size (e.g., 30 to 50 pixels)

print(f"🚀 Running filtered inference on {len(images_list)} images...")

for img_path in images_list:
    # Run prediction in memory without auto-saving the raw messy boxes
    results = model.predict(
        source=img_path,
        imgsz=640,
        rect=True,
        conf=0.1,
        iou=0.30,
        save=False  # We will handle saving manually below
    )
    
    result = results[0]
    img = cv2.imread(img_path)
    
    # Extract bounding boxes data: [x1, y1, x2, y2, confidence, class_id]
    boxes = result.boxes.data.cpu().numpy()
    
    count = 0
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        
        # Calculate width and height of the box
        w = x2 - x1
        h = y2 - y1
        
        # --- THE SIZE FILTER ---
        # Only draw the box if it is smaller than our maximum threshold
        # if w < MAX_PELLET_SIZE and h < MAX_PELLET_SIZE:
        #     count += 1
        #     # Draw a tight, clean bounding box (Blue rectangle, thickness=1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
            
    # Save the processed image to your output directory
    out_name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, out_name)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved {out_name} - Found {count} valid small pellets.")

print(f"🎉 Done! Check your clean images in: {output_dir}")