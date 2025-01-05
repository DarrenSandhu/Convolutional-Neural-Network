import os
import cv2
import numpy as np
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# # device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

# torch.set_default_device(device)


def load_data(animal, label):
    images, labels = [], []
    try:
        # Try loading pre-saved data
        img_file = f"{animal}_images.npy"
        label_file = f"{animal}_labels.npy"
        images = np.load(img_file)
        labels = np.load(label_file)
        print(f"Loaded {animal} data from files.")
    except FileNotFoundError:
        print(f"No pre-saved data found for {animal}. Loading from directory...")
        directory = os.path.join(BASE_DIR, 'training_data', animal)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = os.listdir(directory)
        
        image_size = (64, 64)
        for filename in files[:1000]:
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
        images = np.array(images) / 255.0
        labels = np.array(labels)
        np.save(f"{animal}_images.npy", images)
        np.save(f"{animal}_labels.npy", labels)
        print(f"Saved {animal} data to files.")
    return images, labels

cat_images, cat_labels = load_data('cat', [1, 0])
dog_images, dog_labels = load_data('dog', [0, 1])

# print(f"Cat images shape: {cat_images.shape}, Cat labels shape: {cat_labels.shape}")
# print(f"Dog images shape: {dog_images.shape}, Dog labels shape: {dog_labels.shape}")

images = np.vstack([cat_images, dog_images])
labels = np.vstack([cat_labels, dog_labels])
torch_images = torch.tensor(images, dtype=torch.float32)
torch_labels = torch.tensor(labels, dtype=torch.float32)
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
# print("Labels: ", labels)   

