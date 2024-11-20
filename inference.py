from ultralytics import YOLO

# Load the trained model
model = YOLO(r"runs_ants\feature_extraction8\weights\best.pt")  # Path to the trained model weights

project_path = r'C:\Users\hadar\Documents\database\videos\other\out'
source_path = r'C:\Users\hadar\Documents\database\videos\other\9888614-hd_1920_1080_30fps.mp4'

# Perform inference on a video file
model.predict(
    source=source_path,  # Path to the input video
    imgsz=1280,                   # Input image size
    conf=0.5,                     # Confidence threshold
    save=True,                    # Save annotated predictions
    save_txt=False,               # Disable saving results in YOLO format (optional for video)
    save_crop=False,              # Disable saving cropped images of detected objects (optional)
    project=project_path,  # Directory to save the annotated video
    name='annotated_video',       # Name for the output
    device=0                      # Use GPU (0) or CPU ('cpu')
)


print("Inference complete. Annotated video saved in:", project_path)
