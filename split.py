import cv2
import os

def video_to_frames(video_path, output_folder, frame_interval=20):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    print(f"Extracting every {frame_interval}th frame... Please wait.")
    
    while True:
        # Read the next frame from the video
        success, frame = video_capture.read()
        
        # End of video reached
        if not success:
            break
        
        # Only save if the current frame count is a multiple of your interval
        if frame_count % frame_interval == 0:
            # Format the filename (e.g., frame_0020.jpg, frame_0040.jpg)
            frame_name = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Success! Processed {frame_count} total frames.")
    print(f"Saved {saved_count} pictures to '{output_folder}'.")
# --- HOW TO USE IT ---
# Replace these paths with your actual video file and desired output folder
video_file_path ="192.168.1.64_01_20260522145639814.mp4"
output_directory = "extracted_frames2"

video_to_frames(video_file_path, output_directory, frame_interval=20)