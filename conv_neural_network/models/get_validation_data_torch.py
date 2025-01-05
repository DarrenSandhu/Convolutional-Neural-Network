import os
import cv2
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_device(device)
# # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# # torch.set_default_device(device)

image_size = (64, 64)

def get_validation_data(animal):
    images, labels = [], []
    directory = os.path.join(BASE_DIR, 'validation_data', animal)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = os.listdir(directory)
    for filename in files:
        img_path = os.path.join(directory, filename)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        try:
            if os.path.isfile(img_path) and not img_path.endswith('.jpg'):
                continue
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append([1,0] if animal == 'cat' else [0,1])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    images = np.array(images)
    images = torch.tensor(images) / 255.0  
    labels = torch.tensor(labels)
    return images, labels

cat_images, cat_labels = get_validation_data('cat')
dog_images, dog_labels = get_validation_data('dog')
print(f"Cat validation images shape: {cat_images.shape}, Cat validation labels shape: {cat_labels.shape}")
print(f"Dog validation images shape: {dog_images.shape}, Dog validation labels shape: {dog_labels.shape}")

