from ultralytics import YOLO, solutions
import cv2
import os
from datetime import datetime

import cv2
import numpy as np

def back_shift(frame, prev_frame):
    """
    Aligns the current frame to the previous frame using cv2.cuda.matchTemplate for translation alignment.

    Args:
    - frame (numpy.ndarray): Current frame.
    - prev_frame (numpy.ndarray): Previous frame.

    Returns:
    - numpy.ndarray: The stabilized frame aligned to the previous frame's position.
    """
    if prev_frame is None:
        print("\nPrev_frame is None. If first frame, ignore.")
        return frame

    margin = 200

    # Convert frames to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Crop the frames to reduce edges and focus on central features
    cropped_frame = gray_frame[margin:-margin, margin:-margin]

    # Upload frames to GPU
    d_frame = cv2.cuda_GpuMat()
    d_prev_frame = cv2.cuda_GpuMat()
    d_frame.upload(cropped_frame)
    d_prev_frame.upload(gray_prev_frame)
    
    # Apply template matching using GPU
    match_result = cv2.cuda.createTemplateMatching(cv2.CV_8U, cv2.TM_CCOEFF_NORMED).match(d_prev_frame, d_frame)

    # Download result from GPU
    result = match_result.download()

    # Find the best match location
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate translation
    x_shift = max_loc[0] - margin
    y_shift = max_loc[1] - margin

    # Create translation matrix
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

    # Warp the current frame to align with the previous frame
    aligned_frame = cv2.warpAffine(
        frame,
        translation_matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR
    )

    # Debug info

    print(f"\nx_shift: {x_shift}, y_shift: {y_shift}")
    cv2.imshow("cropped_frame", cv2.resize(cropped_frame, None, fx=0.5, fy=0.5))
    cv2.imshow("gray_prev_frame", cv2.resize(gray_prev_frame, None, fx=0.5, fy=0.5))
    cv2.imshow("gray_frame", cv2.resize(gray_frame, None, fx=0.5, fy=0.5))
    cv2.imshow("aligned_frame", cv2.resize(aligned_frame, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)

    return aligned_frame


# Load the trained model - in the updated version the model is initialized in the class ObjectCounter
# model = YOLO(r"runs_ants\feature_extraction8\weights\best.pt")  # Path to the trained model weights

# Initialize Object Counter, see ObjectCounter.py (ctrl + left click) in the site-packages, where python is installed
counter = solutions.ObjectCounter( 
  view_img=True,                     # Display the image during processing 
#   reg_pts=[(0, 0), (900, 1850)], # Region of interest points 
  region=[(0, 500), (1850, 500)],
#   classes_names=model.names,         # Class names from the YOLO model 
  draw_tracks=True,                  # Draw tracking lines for objects 
  line_thickness=2,                  # Thickness of the lines drawn 
  model=r"runs_ants\feature_extraction8\weights\best.pt" # trained weights for specific objects (see Readme)
  )

# Paths
project_path = r'C:\Users\hadar\Documents\database\videos\other\out'

# source_path = r'C:\Users\hadar\Documents\database\videos\other\9888614-hd_1920_1080_30fps.mp4' # clean vid
source_path = r'C:\Users\hadar\Documents\database\videos\other\output_translated_video.mp4' # video after adding transaltion noise

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
prev_frame = None
# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # old YOLO code (cause the error "object counting dont have attribute start_counting"):

    # results = model.predict(frame, imgsz=1280, conf=0.5, device=0, verbose=False)
    # annotated_frame = results[0].plot()  # Get the annotated frame
    # tracks = model.track(frame, persist=True, tracker='botsort.yaml', iou=0.2) 
    # annotated_frame = counter.initialize_region(frame, tracks) 

    # match template
    to_stable_frame = True
    if to_stable_frame:
        frame = back_shift(frame, prev_frame)
        prev_frame = frame.copy()

    # the updated version YOLO code:
    # Use the Object Counter to count objects in the frame and get the annotated image 
    annotated_frame = counter.count(frame) 
    out.write(annotated_frame)

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

print(f'\nIn: {counter.in_count}\nOut: {counter.out_count}\nTotal: {counter.in_count + counter.out_count}')
print("\nInference complete. Annotated video saved in:", output_video_path)
