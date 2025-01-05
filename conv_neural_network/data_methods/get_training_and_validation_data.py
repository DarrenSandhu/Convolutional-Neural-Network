import os
import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

def load_data(animal, label, split_ratio=0.8):
    """
    Load and preprocess data for the given animal, and split into training and validation sets.
    
    :param animal: The name of the animal (e.g., "cat" or "dog").
    :param label: The label to assign to this animal (e.g., 0 for cat, 1 for dog).
    :param split_ratio: The ratio of data to allocate for training (default: 80% training, 20% validation).
    :return: A tuple (train_data, train_labels, val_data, val_labels).
    """
    images, labels = [], []
    
    try:
        # Try loading pre-saved data
        img_file = f"{animal}_images.pth"
        label_file = f"{animal}_labels.pth"
        images = torch.load(img_file, weights_only=True)
        labels = torch.load(label_file, weights_only=True)
        print(f"Loaded {animal} data from files.")
    except FileNotFoundError:
        print(f"No pre-saved data found for {animal}. Loading from directory...")
        directory = os.path.join(TRAINING_DATA_DIR, animal)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = os.listdir(directory)
        
        image_size = (64, 64)  # Resize all images to this size
        for filename in files:
            img_path = os.path.join(directory, filename)
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Normalize and save processed data
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize pixel values
        labels = torch.tensor(labels, dtype=torch.float32)
        torch.save(images, f"{animal}_images.pth")
        torch.save(labels, f"{animal}_labels.pth")
        print(f"Saved {animal} data to files.")
    
    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=1 - split_ratio, random_state=42
    )
    
    return train_images, train_labels, val_images, val_labels


cat_train_images, cat_train_labels, cat_val_images, cat_val_labels = load_data('cat', [1])
dog_train_images, dog_train_labels, dog_val_images, dog_val_labels = load_data('dog', [0])

print(f"Cat training images shape: {cat_train_images.shape}, Cat training labels shape: {cat_train_labels.shape}")
print(f"Dog training images shape: {dog_train_images.shape}, Dog training labels shape: {dog_train_labels.shape}")

print(f"Cat validation images shape: {cat_val_images.shape}, Cat validation labels shape: {cat_val_labels.shape}")
print(f"Dog validation images shape: {dog_val_images.shape}, Dog validation labels shape: {dog_val_labels.shape}")

# Combine training data for both cats and dogs
train_images = torch.vstack([cat_train_images, dog_train_images])
train_labels = torch.vstack([cat_train_labels, dog_train_labels])

# Combine validation data for both cats and dogs
val_images = torch.vstack([cat_val_images, dog_val_images])
val_labels = torch.vstack([cat_val_labels, dog_val_labels])
