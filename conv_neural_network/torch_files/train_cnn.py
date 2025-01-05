import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_torch import Convolutional_Neural_Network  # Import your custom class
from architecture.conv_layer_list_torch import Convolutional_Layers_Torch
from data_methods.get_cnn_training_data_torch import images, labels

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ",device)
torch.set_default_device(device)

# Create Model Parameters
input_nodes = images.shape[1] * images.shape[2] * images.shape[3] # Number of input nodes equals the number of pixels in an image
print("Images Shape: ",images.shape)
output_nodes = 2
target = labels
print("Target Shape: ",target.shape)
batch_size = 128
#conv_layers = Convolutional_Layers_Torch(2, [[8,5], [16,3]], ['relu', 'relu'], images.shape, batch_size)
# conv_layers = Convolutional_Layers_Torch(2, [[4,5], [8,3]], ['relu', 'relu'], images.shape, batch_size)
conv_layers = Convolutional_Layers_Torch(3, [[16,5], [32,3], [64,3]], ['relu', 'relu', 'relu'], images.shape, batch_size)
# conv_layers = Convolutional_Layers_Torch(4, [[16,5], [32,3], [64,3], [128,3]], ['relu', 'relu', 'relu', 'relu'], images.shape, batch_size)


# Create the model
cnn = Convolutional_Neural_Network(input_nodes, 
                                   output_nodes, 
                                   conv_layers)

# Shorten image list for faster training
images = torch.cat((images[:100], images[1000:1100]), dim=0)
target = torch.cat((target[:100], target[1000:1100]), dim=0)
cnn.train(images, target, 1000, 0.001, batch_size)
cnn.save("cnn_model.pth")