# from ultralytics import YOLO

# def main():
#     # 1. Load a pre-trained YOLOv8 Nano model (yolov8n.pt)
#     # Nano is ideal for laptops without high-end dedicated GPUs and runs fast.
# # New YOLO26 setup
#     model = YOLO("yolo26n.pt")    # 2. Train the model
#     # 2. Train the model
#     results = model.train(
#         data="Fish.v1i.yolo26/data.yaml",  # Updated relative path
#         epochs=100,            
#         imgsz=640,              
#         device="0",            # Forces utilization of your NVIDIA GPU
#         batch=16,               # 8 is the sweet spot for 1024px on laptop VRAM. Push to 16 ONLY if your GPU has 8GB+ VRAM.
#         workers=8,             # Matches your CPU cores to rapidly preprocess images in parallel
#         cache=True,            # CRITICAL: Loads the entire dataset into RAM to eliminate SSD/HDD lag bottlenecks       
#         project="fish_detect1",  
#         name="yolo26n_pellets_v2"  
#     )

# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

if __name__ == '__main__':
# 1. Load a completely fresh, factory-default YOLO26 Small model
    model = YOLO("yolo26s.pt")

    # 2. Start the training pipeline
    results = model.train(
        data="Fish_Pellet.v1i.yolo26/data.yaml",  # Updated to your pellet configuration path
        epochs=100,                      # 100 epochs is a solid starting baseline
        imgsz=640,                       # Standard training image dimension
        batch=16,                        # Adjust based on your available hardware/VRAM
        project="aquaculture_pellet",   # Organizes your results inside a clear folder
        name="pellet_run_s",      # Unique run name for tracking
        resume=False,                    # Guarantees it won't pull from an old checkpoint
        cache=False                      # Forces YOLO26 to re-scan text labels cleanly
    )