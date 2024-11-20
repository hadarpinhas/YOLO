from ultralytics import YOLO
import torch.multiprocessing

def main():
    # Initialize the model with pretrained weights
    model = YOLO('./yolov8s.pt')

    #                   Note:
    #  The downloading of 'yolov11n.pt' or others not used in the model, you might see 
    #  downloaded automatically, is part of a routine check made by the Ultralytics
    #  package to ensure proper functionality of Automatic Mixed Precision (AMP).
    # see https://github.com/ultralytics/ultralytics/issues/4107

    # Train the model with the parameters
    model.train(
        data='data/ants_data.yaml',     # Dataset configuration file
        epochs=150,                     # Number of epochs
        batch=16,                       # Batch size
        cache=True,                     # Cache images for faster training
        freeze=12,                      # Freeze the first 12 layers (backbone)
        project='runs_ants',            # Save results in the "runs_ants" directory
        name='feature_extraction',      # Name of this training run
        device=0,                       # Use GPU (0) or CPU ('cpu')
        workers=torch.multiprocessing.cpu_count() // 2  # Set the number of workers
    )


if __name__ == '__main__':
    # Set the multiprocessing start method (necessary for Windows)
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Run the main function
    main()
