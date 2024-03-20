import os
import random
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the path to your dataset
dataset_path = "./Preprocessed ISL Finger Dataset"

# Create directories for training and test sets
train_path = './Preprocessed ISL Finger Dataset/train'
test_path = './Preprocessed ISL Finger Dataset/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Specify the percentage of data for the test set
test_split = 0.2

# Iterate through each class subfolder
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    # Create subfolders in train and test directories
    train_class_path = os.path.join(train_path, class_folder)
    test_class_path = os.path.join(test_path, class_folder)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # Get a list of images in the class folder
    images = os.listdir(class_path)

    # Shuffle the images to randomize the split
    random.shuffle(images)

    # Calculate the number of images for the test set
    num_test = int(len(images) * test_split)

    # Copy images to training and test sets
    for img in images[:num_test]:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(test_class_path, img)
        copyfile(src_path, dst_path)

    for img in images[num_test:]:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(train_class_path, img)
        copyfile(src_path, dst_path)

# Set up data generators
batch_size = 32
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)