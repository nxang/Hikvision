from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/aquaculture_fish/fish/weights/best.pt")

# Run inference and let YOLO handle the plotting automatically
results = model.predict(source="Captures_2026-05-30/Preset_2_20260530_151643_shot_1.jpg", save=True, conf=0.20)

# results = model.predict(source="Captures_2026-05-26/Preset_2_20260526_151403.jpg", save=True, conf=0.20)
