#Good Start

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_torch_multiple_fc_layers import Convolutional_Neural_Network_Multiple_FC_Layers  # Import your custom class
from architecture.conv_layer_list_torch import Convolutional_Layers_Torch  # Import your custom class
from architecture.fully_connected_layer_list_torch import Fully_Connected_Layers_Torch  # Import your custom class
from data_methods.get_training_and_validation_data import train_images, train_labels, dog_train_images, dog_train_labels, cat_train_images, cat_train_labels, dog_val_images, dog_val_labels, cat_val_images, cat_val_labels

# Traained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ",device)
torch.set_default_device(device)

# Create Model Parameters
input_nodes = train_images.shape[1] * train_images.shape[2] * train_images.shape[3] # Number of input nodes equals the number of pixels in an image
print("Images Shape: ",train_images.shape)
output_nodes = 1
target = train_labels
print("Target Shape: ",target.shape)
batch_size = 256
dropout_ratio = 0.5
lambda_l2 = 0.00001
strides = [1,1,1]
#strides = [1,1]
weight_init = 'kaimeing-uniform-relu'
fc_weight_init = 'kaiming-in'
#conv_layers = Convolutional_Layers_Torch(4, [[8,5], [16,3], [32,3], [64,3]], ['relu', 'relu', 'relu', 'relu'], train_images.shape, batch_size, strides, weight_init)
conv_layers = Convolutional_Layers_Torch(3, [[16,5], [32,3], [64,3]], ['relu', 'relu', 'relu'], train_images.shape, batch_size, strides, weight_init)
#conv_layers = Convolutional_Layers_Torch(2, [[8,5], [16,3]], ['relu', 'relu'], train_images.shape, batch_size, strides, weight_init)

num_features = conv_layers.conv_layers[-1].max_pool_images.view(batch_size, -1).shape[1]
print("Num Features: ",num_features)
#fully_connected_layers = Fully_Connected_Layers_Torch(5, output_sizes=[num_features, num_features*2, num_features, num_features // 2, output_nodes], activation_funcs=['relu', 'relu', 'relu', 'relu', 'sigmoid'])
#fully_connected_layers = Fully_Connected_Layers_Torch(3, output_sizes=[num_features, num_features*2, output_nodes], activation_funcs=['relu', 'relu', 'sigmoid'], weight_init=fc_weight_init)
fully_connected_layers = Fully_Connected_Layers_Torch(4, output_sizes=[num_features, 256, 128, output_nodes], activation_funcs=['relu', 'relu', 'relu', 'sigmoid'], weight_init=fc_weight_init)

cnn = Convolutional_Neural_Network_Multiple_FC_Layers(
    input_nodes, 
    output_nodes,
    conv_layers,
    fully_connected_layers,
    dropout_ratio=dropout_ratio,
    lambda_l2=lambda_l2
    )


# Shorten image list for faster training
learning_rate = 0.0001
# images = torch.cat((train_images[:5000], train_images[12000:17000]), dim=0)
# target = torch.cat((target[:5000], target[12000:17000]), dim=0)
#dog_val_images = dog_val_images[:1000]
#dog_val_labels = dog_val_labels[:1000]
#cat_val_images = cat_val_images[:1000]
#cat_val_labels = cat_val_labels[:1000]
cnn.train(train_images, target, 10000, learning_rate, batch_size, dog_val_images, dog_val_labels, cat_val_images, cat_val_labels)
cnn.save("cnn_model.pth")