from cnn_multiple_images import Convulutional_Neural_Network
import numpy as np
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__name__))

# Load the trained model
def open_trained_np_cnn(filename):
    data = np.load(filename)

    input_nodes = data['input_nodes']
    output_nodes = data['output_nodes']
    print(f"Input nodes: {input_nodes}, Output nodes: {output_nodes}")
    conv_layer_kernels = data['conv_layer_kernels']
    conv_layer_activation_images = data['conv_layer_activation_images']
    conv_layer_max_pool_images = data['conv_layer_max_pool_images']


    conv_layer_2_kernels = data['conv_layer_2_kernels']
    conv_layer_2_activation_images = data['conv_layer_2_activation_images']
    conv_layer_2_max_pool_images = data['conv_layer_2_max_pool_images']

    fully_connected_weights = data['fully_connected_weights']

    bias_output = data['bias_output']
    bias_conv_layer = data['bias_conv_layer']
    bias_conv_layer_2 = data['bias_conv_layer_2']

    cnn = Convulutional_Neural_Network(input_nodes, 
                                       output_nodes, 
                                       conv_layer_kernels, 
                                       conv_layer_activation_images, 
                                       conv_layer_max_pool_images, 
                                       conv_layer_2_activation_images, 
                                       conv_layer_2_kernels, 
                                       conv_layer_2_max_pool_images, 
                                       fully_connected_weights, 
                                       bias_output, 
                                       bias_conv_layer, 
                                       bias_conv_layer_2)
    
    print(f"Model loaded from {filename}")
    return cnn


cnn = open_trained_np_cnn("trained_cnn_model.npz")

# Load the images
cat_directory = os.path.join(BASE_DIR, 'validation_data', 'cat')
cat_directory_files = os.listdir(cat_directory)

images = []
labels = []
image_size = (64, 64)
# Load 1000 cat images
for filename in cat_directory_files:
    img_path = os.path.join(cat_directory, filename)
    assert os.path.exists(img_path), f"File not found: {img_path}"
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append([1, 0])
    except Exception as e:
        # print(f"Error processing image {img_path}: {e}")
        continue  # Skip corrupt images
images = np.array(images) / 255.0
labels = np.array(labels)

# Perform a forward pass
predictions = cnn.predict(images)
print(predictions)
