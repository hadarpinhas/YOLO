import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits YOLO dataset (images and labels) into train, val, and test sets using scikit-learn.

    Parameters:
        images_dir (str): Path to the folder containing images.
        labels_dir (str): Path to the folder containing label files.
        output_dir (str): Path to the output folder.
        train_ratio (float): Proportion of images for training.
        val_ratio (float): Proportion of images for validation.
        test_ratio (float): Proportion of images for testing.

    Returns:
        None
    """

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure that each image has a corresponding label
    image_files = [img for img in image_files if os.path.exists(os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt"))]

    # Split the dataset into train, val, and test
    train_files, temp_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Helper function to copy files to the respective directories
    def copy_files(file_list, split_name):
        for file_name in file_list:
            # Copy image
            src_image = os.path.join(images_dir, file_name)
            dest_image = os.path.join(output_dir, split_name, 'images', file_name)
            shutil.copy(src_image, dest_image)

            # Copy corresponding label
            label_file = os.path.splitext(file_name)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dest_label = os.path.join(output_dir, split_name, 'labels', label_file)
            shutil.copy(src_label, dest_label)

    # Copy files to train, val, and test folders
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset split complete!")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")


# Example Usage
if __name__ == "__main__":
    # Path to the folder containing images
    images_dir = r"C:\Users\hadar\Documents\database\videos\other\ants_dataset\original_data\images"
    # Path to the folder containing labels
    labels_dir = r"C:\Users\hadar\Documents\database\videos\other\ants_dataset\original_data\labels"
    # Path to the output folder
    output_dir = r"C:\Users\hadar\Documents\database\videos\other\ants_dataset\split_data"

    # Split the dataset
    split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
