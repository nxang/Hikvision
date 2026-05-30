import cv2
import numpy as np
import os
import glob

def nothing(x):
    pass

# --- CONFIGURATION PATHS ---
INPUT_FOLDER = 'Captures_2026-05-29'
OUTPUT_FOLDER = 'Captures_2026-05-29_Filtered'
TUNING_IMAGE_PATH = os.path.join(INPUT_FOLDER, 'Preset_11_20260529_124022.jpg')
# 1. Load the target tuning image
image = cv2.imread(TUNING_IMAGE_PATH)
if image is None:
    print(f"Error: Could not load tuning image at {TUNING_IMAGE_PATH}")
    exit()

# Convert to HSV for initial color tracking
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. Create the tuning GUI window
window_name = 'Raw HSV Mask Tuner'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

# 3. Initialize all trackbars with your custom values
cv2.createTrackbar('Min H', window_name, 13, 180, nothing)
cv2.createTrackbar('Max H', window_name, 45, 180, nothing)
cv2.createTrackbar('Min S', window_name, 10, 255, nothing)
cv2.createTrackbar('Max S', window_name, 255, 255, nothing)
cv2.createTrackbar('Min V', window_name, 88, 255, nothing)
cv2.createTrackbar('Max V', window_name, 255, 255, nothing)
cv2.createTrackbar('Highlight Opacity', window_name, 100, 100, nothing)

cv2.createTrackbar('Min Size (Area)', window_name, 2, 150, nothing)
cv2.createTrackbar('Max Size (Area)', window_name, 2000, 2000, nothing)
cv2.createTrackbar('Min Roundness (%)', window_name, 7, 100, nothing)
cv2.createTrackbar('Min Yellow Contrast', window_name, 4, 100, nothing)

# Windows for real-time validation feedback
cv2.namedWindow('Live Binary Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Live Highlighted Content', cv2.WINDOW_NORMAL)

# Define static color matrix layer (Neon Green)
color_layer = np.zeros_like(image)
color_layer[:] = [0, 255, 0] 

print("Tuning environment ready. Press 'ESC' to lock values and start folder batch processing.")

# Active storage dictionary for configuration properties
final_params = {}

while True:
    # Read dynamic slide inputs
    min_h = cv2.getTrackbarPos('Min H', window_name)
    max_h = cv2.getTrackbarPos('Max H', window_name)
    min_s = cv2.getTrackbarPos('Min S', window_name)
    max_s = cv2.getTrackbarPos('Max S', window_name)
    min_v = cv2.getTrackbarPos('Min V', window_name)
    max_v = cv2.getTrackbarPos('Max V', window_name)
    opacity = cv2.getTrackbarPos('Highlight Opacity', window_name) / 100.0
    
    min_size = cv2.getTrackbarPos('Min Size (Area)', window_name)
    max_size = cv2.getTrackbarPos('Max Size (Area)', window_name)
    min_circularity = cv2.getTrackbarPos('Min Roundness (%)', window_name) / 100.0
    yellow_contrast_thresh = cv2.getTrackbarPos('Min Yellow Contrast', window_name)

    # Core Pipeline Logic Step 1: Color Masking
    lower_bounds = np.array([min_h, min_s, min_v])
    upper_bounds = np.array([max_h, max_s, max_v])
    raw_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    # Core Pipeline Logic Step 2: Morphological Opening Noise Filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # Core Pipeline Logic Step 3: Geometry & BGR Color Verification
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(cleaned_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_size <= area <= max_size:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity >= min_circularity:
                    # Isolate target contour for specialized average check
                    c_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
                    cv2.drawContours(c_mask, [contour], -1, 255, -1)
                    
                    mean_val = cv2.mean(image, mask=c_mask)
                    b_avg, g_avg, r_avg = mean_val[0], mean_val[1], mean_val[2]
                    
                    r_b_difference = r_avg - b_avg
                    g_b_difference = g_avg - b_avg
                    
                    if r_b_difference >= yellow_contrast_thresh and g_b_difference >= yellow_contrast_thresh:
                        cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    # Generate Render Layer Overlay
    blended_layer = cv2.addWeighted(image, 1.0 - opacity, color_layer, opacity, 0)
    highlighted_result = np.where(filtered_mask[:, :, None] == 255, blended_layer, image)

    # Display real-time video matrices
    cv2.imshow('Live Binary Mask', filtered_mask)
    cv2.imshow('Live Highlighted Content', highlighted_result)

    # Break processing on Escape key registration
    if cv2.waitKey(1) & 0xFF == 27:
        final_params = {
            'lower': lower_bounds, 'upper': upper_bounds, 'opacity': opacity,
            'min_size': min_size, 'max_size': max_size, 'min_circ': min_circularity,
            'contrast_thresh': yellow_contrast_thresh
        }
        break

cv2.destroyAllWindows()


# --- STARTING THE BATCH FOLDER PROCESSING LAYER ---
print(f"\nProcessing active files within path: '{INPUT_FOLDER}'...")

# Instantiate output directory structures if unavailable
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created tracking output folder at: '{OUTPUT_FOLDER}'")

# Look up images within folder space
valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_files = []
for ext in valid_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

if not image_files:
    print(f"Error: No image assets detected inside directory target: '{INPUT_FOLDER}'. Closing execution loop.")
    exit()

print(f"Successfully tracked {len(image_files)} image components. Compiling filter parameters...")

# Re-run clean logic across files matching configuration targets precisely
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping corrupted/unreadable asset target: {img_path}")
        continue
        
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Process identical pipeline sequence
    r_mask = cv2.inRange(img_hsv, final_params['lower'], final_params['upper'])
    
    proc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    c_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN, proc_kernel)
    
    cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_mask = np.zeros_like(c_mask)
    
    for c in cnts:
        a = cv2.contourArea(c)
        if final_params['min_size'] <= a <= final_params['max_size']:
            p = cv2.arcLength(c, True)
            if p > 0:
                circ = (4 * np.pi * a) / (p ** 2)
                if circ >= final_params['min_circ']:
                    
                    # Individual pixel validation tracking lookup matching tuner logic
                    single_c_mask = np.zeros(c_mask.shape, dtype=np.uint8)
                    cv2.drawContours(single_c_mask, [c], -1, 255, -1)
                    
                    m_val = cv2.mean(img, mask=single_c_mask)
                    b_a, g_a, r_a = m_val[0], m_val[1], m_val[2]
                    
                    diff_rb = r_a - b_a
                    diff_gb = g_a - b_a
                    
                    if diff_rb >= final_params['contrast_thresh'] and diff_gb >= final_params['contrast_thresh']:
                        cv2.drawContours(f_mask, [c], -1, 255, -1)
                        
    # Draw non-destructive overlay highlights onto current index matrix file
    batch_color_layer = np.zeros_like(img)
    batch_color_layer[:] = [0, 255, 0]
    
    b_layer = cv2.addWeighted(img, 1.0 - final_params['opacity'], batch_color_layer, final_params['opacity'], 0)
    final_output = np.where(f_mask[:, :, None] == 255, b_layer, img)
    
    # Save step matching structural formats
    file_base = os.path.basename(img_path)
    output_filename = os.path.join(OUTPUT_FOLDER, f"highlighted_{file_base}")
    
    cv2.imwrite(output_filename, final_output)
    print(f"Processed & Written to Disk: {output_filename}")

print(f"\nExecution Complete! All glare-filtered results successfully stored in folder: '{OUTPUT_FOLDER}'")