import torch
import os
import cv2
import sys
import numpy as np
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
print(f"Path scripts: {path_scripts}")
sys.path.append(str(path_scripts))

from architecture.conv_layer_list_torch import Convolutional_Layers_Torch  # Import your custom class
from architecture.conv_layer_torch import Convolution_Layer_Torch  # Import your custom class
from architecture.fully_connected_layer_list_torch import Fully_Connected_Layers_Torch  # Import your custom class
from architecture.fully_connected_layer_torch import Fully_Connected_Layer  # Import your custom class
from torch_files.cnn_torch import Convolutional_Neural_Network  # Import your custom class
from torch_files.cnn_torch_multiple_fc_layers import Convolutional_Neural_Network_Multiple_FC_Layers  # Import your custom class

torch.serialization.add_safe_globals([Convolutional_Layers_Torch])
torch.serialization.add_safe_globals([Convolution_Layer_Torch])
torch.serialization.add_safe_globals([Fully_Connected_Layers_Torch])
torch.serialization.add_safe_globals([Fully_Connected_Layer])

BASE_DIR = os.path.dirname(os.path.abspath(__name__))

# Load the trained model
def open_trained_torch_cnn(filename):
    data = torch.load(filename, weights_only=True, map_location='cpu')

    # Model parameters
    input_nodes = data['input_nodes']
    # print(f"Input nodes: {input_nodes}")
    output_nodes = data['output_nodes']
    # print(f"Output nodes: {output_nodes}")
    bias_output = data['bias_output']
    # print(f"Bias output Shape: {bias_output.shape}")
    bias_conv_layers = data['bias_conv_layers']
    # print(f"Bias conv layers Shape: {len(bias_conv_layers)}")
    conv_layers = data['conv_layers']
    # print(f"Conv layers Shape: {conv_layers}")
    fc_layers = data['fc_layers']

    # Create the model
    cnn = Convolutional_Neural_Network(input_nodes, output_nodes, conv_layers, fc_layers, bias_output, bias_conv_layers)
    print(f"Model loaded from {filename}")
    return cnn

def open_trained_torch_multiple_fc_cnn(filename):
    data = torch.load(filename, weights_only=True, map_location='cpu')

    # Model parameters
    input_nodes = data['input_nodes']
    # print(f"Input nodes: {input_nodes}")
    output_nodes = data['output_nodes']
    # print(f"Output nodes: {output_nodes}")
    fc_layers = data['fc_layers']
    # print(f"Fully connected weights Shape: {fully_connected_weights.shape}")
    bias_output = data['bias_output']
    # print(f"Bias output Shape: {bias_output.shape}")
    bias_conv_layers = data['bias_conv_layers']
    # print(f"Bias conv layers Shape: {len(bias_conv_layers)}")
    conv_layers = data['conv_layers']
    # print(f"Conv layers Shape: {conv_layers}")

    # Create the model
    cnn = Convolutional_Neural_Network_Multiple_FC_Layers(input_nodes, output_nodes, conv_layers, fc_layers, bias_output, bias_conv_layers)
    print(f"Model loaded from {filename}")
    return cnn


# filename = "real_cnn_model.pth" 
# cnn = open_trained_torch_multiple_fc_cnn(filename) 
# batch_size = cnn.conv_layers[0].activation_images.shape[0]





# dog_label = 0
# cat_label = 1

# print("---------------------")
# # Load a test image
# image_size = (64, 64)

# test_image = cv2.imread("/Users/darrensandhu/Projects/AI/ArtificialNeuralNetwork/dog_test_image.jpeg")
# test_image_2 = cv2.imread("/Users/darrensandhu/Projects/AI/ArtificialNeuralNetwork/cat_test_image.jpg")
# test_image_3 = cv2.imread("/Users/darrensandhu/Projects/AI/ArtificialNeuralNetwork/cat_test_image_2.jpeg")
# test_image_4 = cv2.imread("/Users/darrensandhu/Projects/AI/ArtificialNeuralNetwork/Cat03.jpg")
# test_image_5 = cv2.imread("/Users/darrensandhu/Projects/AI/ArtificialNeuralNetwork/conv_neural_network/training_data/cat/9.jpg")

# test_image = cv2.resize(test_image, image_size)
# test_image_2 = cv2.resize(test_image_2, image_size)
# test_image_3 = cv2.resize(test_image_3, image_size)
# test_image_4 = cv2.resize(test_image_4, image_size)
# test_image_5 = cv2.resize(test_image_5, image_size)

# images = []
# images.append(test_image)
# images.append(test_image_2)
# images.append(test_image_3)
# images.append(test_image_4)
# images.append(test_image_5)

# images = np.array(test_image_5)
# image = (torch.tensor(images) / 255.0)
# print(f"Image shape: {image.shape}")

# # for i in range(image.shape[0]):
# prediction = cnn.predict_single_image(image.unsqueeze(0), batch_size).item()
# print(f"Prediction: {prediction}")
# if prediction < 0.5:
#     print("Dog")
# else:
#     print("Cat")

# if prediction < 0.5:
#     print("Dog")
