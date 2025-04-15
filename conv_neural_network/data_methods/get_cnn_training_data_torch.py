import os
import cv2
import torch
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)


def load_data(animal, label, split=0.8):
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
        
        image_size = (64, 64)
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
        images = torch.tensor(images) / 255.0
        labels = torch.tensor(labels)
        torch.save(images, f"{animal}_images.pth")
        torch.save(labels, f"{animal}_labels.pth")
        print(f"Saved {animal} data to files.")
    return images, labels

cat_images, cat_labels = load_data('cat', [1])
dog_images, dog_labels = load_data('dog', [0])

print(f"Cat images shape: {cat_images.shape}, Cat labels shape: {cat_labels.shape}")

# images = torch.vstack([cat_images, dog_images])
# labels = torch.vstack([cat_labels, dog_labels])
images = torch.vstack([cat_images, dog_images]).to(device)
labels = torch.vstack([cat_labels, dog_labels]).to(device)
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
