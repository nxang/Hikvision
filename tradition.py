import cv2
import numpy as np

def nothing(x):
    pass

# 1. Load the original image
# image = cv2.imread('Captures_2026-05-29/Preset_10_20260529_123743.jpg')
image = cv2.imread('Captures_2026-05-29/Preset_11_20260529_124022.jpg')
# image = cv2.imread('Captures_2026-05-29/Preset_7_20260529_130026.jpg')


if image is None:
    print("Error: Could not load image.")
    exit()
    
# Feel free to uncomment this line if you want to activate the bubble blur step:
# image = cv2.medianBlur(image, 5)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
cv2.createTrackbar('Max Size (Area)', window_name, 2000, 2000, nothing)

# NEW: Circularity slider (0% = completely irregular shape, 100% = perfect mathematical circle)
# Initialized to 55% as a great baseline to eliminate glare streaks
cv2.createTrackbar('Min Roundness (%)', window_name, 18, 100, nothing)

# Set up display windows for the outputs
cv2.namedWindow('Live Binary Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Live Highlighted Content', cv2.WINDOW_NORMAL)

# 4. Define the Highlight Color (Neon Green)
color_layer = np.zeros_like(image)
color_layer[:] = [0, 255, 0] 

print("Adjust the sliders. Press 'ESC' to close.")

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
    
    # Read roundness percentage and convert it to a 0.0 - 1.0 float threshold
    min_circularity = cv2.getTrackbarPos('Min Roundness (%)', window_name) / 100.0

    # Apply the raw color mask
    lower_bounds = np.array([min_h, min_s, min_v])
    upper_bounds = np.array([max_h, max_s, max_v])
    raw_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    # 5. Size AND Shape (Circularity) Filtering via Contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # Change raw_mask to cleaned_mask in the line below
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(cleaned_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Step A: Filter by Area size limits
        if min_size <= area <= max_size:
            # Step B: Calculate perimeter (arc length)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                # Step C: Run Circularity Formula
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                # Step D: Drop highlight if shape isn't circular enough
                if circularity >= min_circularity:
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
        print(f"\nYour Tuned Configuration:")
        print(f"Lower HSV: [{min_h}, {min_s}, {min_v}]")
        print(f"Upper HSV: [{max_h}, {max_s}, {max_v}]")
        print(f"Allowed Area Size: {min_size} to {max_size} pixels")
        print(f"Min Roundness Cutoff: {min_circularity * 100:.0f}%")
        break

cv2.destroyAllWindows()