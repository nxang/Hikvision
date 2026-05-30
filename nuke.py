import os
import glob

# Path to your active dataset folder
DATASET_DIR = r"C:\Users\yieku\Desktop\Hikvision\Fish.v5i.yolo26"

# Find every text label file
txt_files = glob.glob(os.path.join(DATASET_DIR, "**", "labels", "*.txt"), recursive=True)
removed_count = 0
modified_files = 0

for file_path in txt_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    clean_lines = []
    file_changed = False
    
    for line in lines:
        # A valid object detection box MUST have exactly 5 elements
        if len(line.strip().split()) == 5:
            clean_lines.append(line)
        else:
            removed_count += 1
            file_changed = True
            
    if file_changed:
        with open(file_path, 'w') as f:
            f.writelines(clean_lines)
        modified_files += 1

print(f"Success! Permanently deleted {removed_count} polygon lines across {modified_files} files.")