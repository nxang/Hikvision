import cv2
import numpy as np
import os
import glob

def nothing(x):
    pass

# --- CONFIGURATION ---
INPUT_FOLDER = 'Captures_2026-05-29'
OUTPUT_FOLDER = 'Captures_2026-05-29_Filtered'
TUNING_IMAGE_PATH = os.path.join(INPUT_FOLDER, 'Preset_11_20260529_123754.jpg')

# 1. Load the tuning image
image = cv2.imread(TUNING_IMAGE_PATH)
if image is None:
    print(f"Error: Could not load tuning image at {TUNING_IMAGE_PATH}")
    exit()
    
bubble_free_image = cv2.medianBlur(image, 5)
hsv = cv2.cvtColor(bubble_free_image, cv2.COLOR_BGR2HSV)

# 2. Create the tuning window
window_name = 'Raw HSV Mask Tuner'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

# 3. Create HSV sliders + Transparency slider
cv2.createTrackbar('Min H', window_name, 13, 180, nothing)
cv2.createTrackbar('Max H', window_name, 45, 180, nothing)
cv2.createTrackbar('Min S', window_name, 10, 255, nothing)
cv2.createTrackbar('Max S', window_name, 255, 255, nothing)
cv2.createTrackbar('Min V', window_name, 88, 255, nothing)
cv2.createTrackbar('Max V', window_name, 255, 255, nothing)
cv2.createTrackbar('Highlight Opacity', window_name, 100, 100, nothing)

# Size sliders (measured in total pixel area)
cv2.createTrackbar('Min Size (Area)', window_name, 2, 150, nothing)
cv2.createTrackbar('Max Size (Area)', window_name, 1800, 2000, nothing)

# Set up display windows for the outputs
cv2.namedWindow('Live Binary Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Live Highlighted Content', cv2.WINDOW_NORMAL)

# 4. Define the Highlight Color (Neon Green)
color_layer = np.zeros_like(image)
color_layer[:] = [0, 255, 0] 

print("Adjust the sliders to perfect your settings. Press 'ESC' to lock them in and batch-process the entire folder.")

# Variables to store final tuned settings
final_params = {}

while True:
    # Get current slider positions
    min_h = cv2.getTrackbarPos('Min H', window_name)
    max_h = cv2.getTrackbarPos('Max H', window_name)
    min_s = cv2.getTrackbarPos('Min S', window_name)
    max_s = cv2.getTrackbarPos('Max S', window_name)
    min_v = cv2.getTrackbarPos('Min V', window_name)
    max_v = cv2.getTrackbarPos('Max V', window_name)
    opacity = cv2.getTrackbarPos('Highlight Opacity', window_name) / 100.0
    
    min_size = cv2.getTrackbarPos('Min Size (Area)', window_name)
    max_size = cv2.getTrackbarPos('Max Size (Area)', window_name)

    # Apply the raw color mask
    lower_bounds = np.array([min_h, min_s, min_v])
    upper_bounds = np.array([max_h, max_s, max_v])
    raw_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    # 5. Size Filtering via Contours
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(raw_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_size <= area <= max_size:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    # 6. Create the translucent blend layer
    blended_layer = cv2.addWeighted(image, 1.0 - opacity, color_layer, opacity, 0)

    # 7. Apply the overlay using our size-restricted mask
    highlighted_result = np.where(filtered_mask[:, :, None] == 255, blended_layer, image)

    # Show outputs
    cv2.imshow('Live Binary Mask', filtered_mask)
    cv2.imshow('Live Highlighted Content', highlighted_result)

    # Break loop if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        # Store the tuned parameters before closing the UI
        final_params = {
            'lower': lower_bounds, 'upper': upper_bounds, 'opacity': opacity,
            'min_size': min_size, 'max_size': max_size
        }
        print(f"\n--- Settings Locked In ---")
        print(f"Lower HSV: {lower_bounds} | Upper HSV: {upper_bounds}")
        print(f"Area limits: {min_size} to {max_size} pixels")
        break

cv2.destroyAllWindows()


# --- AUTOMATED BATCH PROCESSING SECTION ---
print(f"\nStarting batch processing on files inside folder: '{INPUT_FOLDER}'...")

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: '{OUTPUT_FOLDER}'")

# Find all common image formats inside the folder
valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_files = []
for ext in valid_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

if not image_files:
    print(f"No images found in '{INPUT_FOLDER}'. Execution stopped.")
    exit()

print(f"Found {len(image_files)} images to process. Please wait...")

# Process each image with the final_params
for img_path in image_files:
    # Read the current loop image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable file: {img_path}")
        continue
        
    # Process steps matching the interactive setup exactly
    b_free = cv2.medianBlur(img, 5)
    img_hsv = cv2.cvtColor(b_free, cv2.COLOR_BGR2HSV)
    
    # Apply color mask
    r_mask = cv2.inRange(img_hsv, final_params['lower'], final_params['upper'])
    
    # Filter by size
    cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_mask = np.zeros_like(r_mask)
    
    for c in cnts:
        a = cv2.contourArea(c)
        if final_params['min_size'] <= a <= final_params['max_size']:
            cv2.drawContours(f_mask, [c], -1, 255, -1)
            
    # Apply Highlight layer
    c_layer = np.zeros_like(img)
    c_layer[:] = [0, 255, 0] # Neon Green
    b_layer = cv2.addWeighted(img, 1.0 - final_params['opacity'], c_layer, final_params['opacity'], 0)
    final_output = np.where(f_mask[:, :, None] == 255, b_layer, img)
    
    # Generate the output filename
    base_name = os.path.basename(img_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"highlighted_{base_name}")
    
    # Save to disk
    cv2.imwrite(output_path, final_output)
    print(f"Saved: {output_path}")

print(f"\nSuccess! All highlighted images are saved in: '{OUTPUT_FOLDER}'")