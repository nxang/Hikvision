import os
import glob

# Pre-configured with your exact fish food pellet dataset path
PELLET_DATASET_DIR = r"C:\Users\yieku\Desktop\Hikvision\Fish_Pellet.v1i.yolo26" 

def polygon_to_bbox(line):
    parts = line.strip().split()
    
    # If it already has exactly 5 elements, it's a perfect bounding box. Keep it!
    if len(parts) == 5:
        return line
    
    # If it has more than 5 elements, it's a polygon line
    if len(parts) > 5:
        class_id = parts[0]
        coordinates = [float(x) for x in parts[1:]]
        
        # Split into separate X and Y lists (alternating values)
        x_coords = coordinates[0::2]
        y_coords = coordinates[1::2]
        
        # Find the outermost boundaries of the polygon shape
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # Calculate YOLO format (center_x, center_y, width, height)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        
        # Return the clean 5-column bounding box string
        return f"{class_id} {cx:.7f} {cy:.7f} {w:.7f} {h:.7f}\n"
    
    return None

# Find all .txt files in the train and val label folders
label_files = glob.glob(os.path.join(PELLET_DATASET_DIR, "**", "labels", "*.txt"), recursive=True)

print(f"Scanning text files in: {PELLET_DATASET_DIR}")
converted_lines_count = 0
updated_files_count = 0

for file_path in label_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    repaired_lines = []
    file_was_changed = False
    
    for line in lines:
        repaired_line = polygon_to_bbox(line)
        if repaired_line:
            if repaired_line != line:
                file_was_changed = True
                converted_lines_count += 1
            repaired_lines.append(repaired_line)
            
    if file_was_changed:
        with open(file_path, 'w') as f:
            f.writelines(repaired_lines)
        updated_files_count += 1

print("\n--- Processing Complete ---")
print(f"Successfully converted {converted_lines_count} pellet polygon lines into perfect rectangles!")
print(f"Repaired a total of {updated_files_count} text files.")