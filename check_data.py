import glob

label_files = glob.glob(r"C:\Users\hadar\Documents\database\videos\other\ants_dataset\original_data\labels/*.txt")
for file in label_files[:100]:
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_index = int(line.split()[0])
            if class_index < 0 or class_index >= 1:  # Replace 1 with your `nc`
                print(f"Invalid class index {class_index} in file {file}")