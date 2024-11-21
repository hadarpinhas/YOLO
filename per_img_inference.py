from ultralytics import YOLO, solutions
import cv2
import os
from datetime import datetime

# Load the trained model
# model = YOLO(r"runs_ants\feature_extraction8\weights\best.pt")  # Path to the trained model weights

# Initialize Object Counter
counter = solutions.ObjectCounter( 
  view_img=True,                     # Display the image during processing 
  reg_pts=[(0, 0), (900, 1850)], # Region of interest points 
#   classes_names=model.names,         # Class names from the YOLO model 
  draw_tracks=True,                  # Draw tracking lines for objects 
  line_thickness=2,                  # Thickness of the lines drawn 
  model=r"runs_ants\feature_extraction8\weights\best.pt"
  )

# Paths
project_path = r'C:\Users\hadar\Documents\database\videos\other\out'
# source_path = r'C:\Users\hadar\Documents\database\videos\other\9888614-hd_1920_1080_30fps.mp4'
source_path = r'C:\Users\hadar\Documents\database\videos\other\output_translated_video.mp4'


# Create a timestamp
timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

# Output video path with timestamp
output_video_path = os.path.join(project_path, f'annotated_video_{timestamp}.mp4')

# Create output directory if it doesn't exist
os.makedirs(project_path, exist_ok=True)

# Open the video source
cap = cv2.VideoCapture(source_path)

# Get video properties
w, h, fps, frame_count = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_index = 0

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    # results = model.predict(frame, imgsz=1280, conf=0.5, device=0, verbose=False)
    # annotated_frame = results[0].plot()  # Get the annotated frame

      # Perform object tracking on the current frame 
    # tracks = model.track(frame, persist=True, tracker='botsort.yaml', iou=0.2) 
    # Use the Object Counter to count objects in the frame and get the annotated image 
    annotated_frame = counter.count(frame) 

    out.write(frame)

    # Show progress
    frame_index += 1
    print(f"Processed frame {frame_index}/{frame_count}", end="\r")

    # Optionally display the frame (press 'q' to exit early)
    cv2.imshow("Annotated Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'In: {counter.in_count}\nOut: {counter.out_count}\nTotal: {counter.in_count + counter.out_count}')
print("\nInference complete. Annotated video saved in:", output_video_path)
