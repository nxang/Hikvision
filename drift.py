import cv2
import numpy as np

# --- 1. ENTER YOUR LOCKED TUNED SLIDER SETTINGS HERE ---
LOWER_HSV = np.array([13, 10, 88])
UPPER_HSV = np.array([45, 255, 255])
MIN_SIZE = 2
MAX_SIZE = 2000
MIN_CIRCULARITY = 0.07          # 7% from your trackbar
YELLOW_CONTRAST_THRESH = 22     # 22 from your trackbar
HIGHLIGHT_OPACITY = 1.0         # 100% opacity

def extract_candidate_mask(image):
    """Processes a single frame using your tuned pipeline to find candidate blobs."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_mask = np.zeros_like(cleaned_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_SIZE <= area <= MAX_SIZE:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity >= MIN_CIRCULARITY:
                    
                    # Internal BGR check
                    c_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
                    cv2.drawContours(c_mask, [contour], -1, 255, -1)
                    
                    mean_val = cv2.mean(image, mask=c_mask)
                    b_avg, g_avg, r_avg = mean_val[0], mean_val[1], mean_val[2]
                    
                    if (r_avg - b_avg) >= YELLOW_CONTRAST_THRESH and (g_avg - b_avg) >= YELLOW_CONTRAST_THRESH:
                        cv2.drawContours(output_mask, [contour], -1, 255, -1)
    return output_mask

# --- 2. LOAD YOUR CONSECUTIVE 1-SECOND SHOTS ---
frame1 = cv2.imread('Captures_2026-05-29/Preset_9_20260529_153442_shot_1.jpg')
frame2 = cv2.imread('Captures_2026-05-29/Preset_9_20260529_153443_shot_2.jpg')
frame3 = cv2.imread('Captures_2026-05-29/Preset_9_20260529_153445_shot_3.jpg')

if frame1 is None or frame2 is None or frame3 is None:
    print("Error: Could not load all three shots. Verify filenames.")
    exit()

# --- 3. GENERATE SEPARATE MASKS ---
print("Extracting features from individual frames...")
mask1 = extract_candidate_mask(frame1)
mask2 = extract_candidate_mask(frame2)
mask3 = extract_candidate_mask(frame3)

# --- 4. APPLY TEMPORAL DRIFT TOLERANCE ---
# Dilate the masks to create a 9x9 pixel boundary envelope. 
# This handles the 1-second drift distance of floating pellets.
drift_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dilated_mask1 = cv2.dilate(mask1, drift_kernel)
dilated_mask2 = cv2.dilate(mask2, drift_kernel)
dilated_mask3 = cv2.dilate(mask3, drift_kernel)

# Intersect the expanded envelopes. 
# An item is kept ONLY if it existed consistently across all three seconds.
temporal_intersection = cv2.bitwise_and(dilated_mask1, dilated_mask2)
temporal_intersection = cv2.bitwise_and(temporal_intersection, dilated_mask3)

# Map the validated areas back onto the original frame 1 mask 
# This shrinks the bloated windows back down to precise pellet shapes
final_pellet_mask = cv2.bitwise_and(mask1, temporal_intersection)

# --- 5. RENDER THE HIGHLIGHT OVERLAY ---
color_layer = np.zeros_like(frame1)
color_layer[:] = [0, 255, 0] # Neon Green

blended_layer = cv2.addWeighted(frame1, 1.0 - HIGHLIGHT_OPACITY, color_layer, HIGHLIGHT_OPACITY, 0)
highlighted_result = np.where(final_pellet_mask[:, :, None] == 255, blended_layer, frame1)

# Display Results
cv2.namedWindow('Validated Pellets (Glare Erased)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Validated Pellets (Glare Erased)', 1280, 720)
cv2.imshow('Validated Pellets (Glare Erased)', highlighted_result)

print("Processing complete. Glare successfully filtered out via temporal drift validation.")
cv2.waitKey(0)
cv2.destroyAllWindows()