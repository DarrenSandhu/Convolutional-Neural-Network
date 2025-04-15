import torch
import torch.nn.functional as F
import time
import math
import numpy as np
from torch import nn
import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__name__))

# # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: ",device)
# torch.set_default_device(device)
# print("Default Device: ",torch.get_default_device())

class Convolutional_Neural_Network:
    def __init__(self, input_nodes, output_nodes, conv_layers, fully_connected_weights=None, bias_output=None, bias_conv_layers=None):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        print("Input Nodes: ", self.input_nodes)
        print("Output Nodes: ", self.output_nodes)
        self.conv_layers = conv_layers
        self.last_conv_layer = conv_layers.getLastLayer()
        
        # Initialize weights and biases for the fully connected layer
        if fully_connected_weights is not None:
            self.fully_connected_weights = fully_connected_weights
        else:
            in_features = (self.last_conv_layer.max_pool_images.shape[1] * 
                           self.last_conv_layer.max_pool_images.shape[2] * 
                           self.last_conv_layer.max_pool_images.shape[3])
            self.fully_connected_weights = torch.randn(in_features, self.output_nodes) * torch.sqrt(torch.tensor(2.0 / in_features + self.output_nodes))
            torch.nn.init.kaiming_uniform_(self.fully_connected_weights, a=math.sqrt(5.0))
            print("Fully Connected Weights: ",self.fully_connected_weights)
        
        print("Fully Connected Weights Shape: ", self.fully_connected_weights.shape)
        
        if bias_output is not None:
            self.bias_output = bias_output
        else:
            self.bias_output = torch.randn(1, self.output_nodes) * 0.01
        
        print("Output Bias Shape: ", self.bias_output.shape)

        # Initialize biases for all convolutional layers
        self.bias_conv_layers = []
        print("Conv Layers Size: ", self.conv_layers.size())
        for i in range(self.conv_layers.size()):
            kernel_shape = self.conv_layers.getLayer(i).kernels.shape
            self.bias_conv_layers.append(torch.zeros(1, kernel_shape[0]))
        
        for i, bias in enumerate(self.bias_conv_layers):
            print(f"Bias Conv Layer {i} Shape: ", bias.shape)

    def relu(self, x):
        return torch.relu(x)

    def sigmoid(self, z_value):
        return torch.sigmoid(z_value)
    
    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def mean_squared_error(self, target_output, output):
        return torch.mean(torch.square(target_output - output))

    def softmax(self, x):
        return F.softmax(x, dim=1)

    def softmax_derivative(self, output):
        s = self.softmax(output)
        return s * (1 - s)

    def cross_entropy(self, target_output, output, epsilon=1e-7):
        output = torch.clamp(output, epsilon, 1 - epsilon)
        return -torch.mean(target_output * torch.log(output) + (1 - target_output) * torch.log(1 - output))

    def calculate_delta(self, error, derivative):
        return error * derivative

    def calculate_loss(self, y_true, y_pred):
        epsilon = 1e-15  # Avoid log(0)
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss

    def calculate_error(self, target_output, final_output):
        return final_output - target_output
    
    def convolve_batch_conv2d(self, X_batch, kernels, biases, activation_images, activation_func, stride):
        """
        Convolves each image in the batch `X_batch` with the given `kernels` and `biases`,
        and stores the result in `activation_images` after applying the specified `activation_func`.
        Optimized for CPU without using f.conv2d.
        
        X_batch: Input batch of images (batch_size, height, width, channels)
        kernels: Convolutional kernels (num_kernels, kernel_height, kernel_width, num_input_channels)
        biases: Bias terms for the convolutional layer (num_kernels,)
        activation_images: Output activation images (num_kernels, height, width)
        activation_func: Activation function ('relu', etc.)
        stride: Stride used in the convolution operation
        """
        
        # Ensure X_batch is in (batch_size, channels, height, width)
        X_batch = X_batch.permute(0, 3, 1, 2)  # Convert to (batch_size, channels, height, width)
        
        # Reshape kernels to (out_channels, in_channels, kernel_height, kernel_width)
        reshaped_kernels = kernels.permute(0, 3, 1, 2)
        kernel_size = reshaped_kernels.shape[2]

        biases = biases.squeeze(0)

        reshaped_output = F.conv2d(X_batch, reshaped_kernels, bias=biases, stride=stride)
        activation_images[:] = torch.relu(reshaped_output).permute(0, 2, 3, 1) + biases
        return activation_images

    def convolve_batch(self, X_batch, kernels, biases, activation_images, activation_func, stride):
        """
        Convolves each image in the batch `X_batch` with the given `kernels` and `biases`,
        and stores the result in `activation_images` after applying the specified `activation_func`.
        Optimized for CPU without using f.conv2d.
        
        X_batch: Input batch of images (batch_size, height, width, channels)
        kernels: Convolutional kernels (num_kernels, kernel_height, kernel_width, num_input_channels)
        biases: Bias terms for the convolutional layer (num_kernels,)
        activation_images: Output activation images (num_kernels, height, width)
        activation_func: Activation function ('relu', etc.)
        stride: Stride used in the convolution operation
        """
        # Custom implementation
        # start_time = time.time()

        batch_size, in_height, in_width, in_channels = X_batch.shape
        out_channels, kernel_height, kernel_width, num_input_channels = kernels.shape
        out_height = (in_height - kernel_height) // stride + 1
        out_width = (in_width - kernel_width) // stride + 1

        X_unfolded = X_batch.unfold(1, kernel_height, stride).unfold(2, kernel_width, stride)
        X_unfolded = X_unfolded.permute(0, 1, 2, 4, 5, 3).reshape(batch_size, out_height * out_width, -1)
        reshaped_kernels = kernels.reshape(out_channels, -1).t()
        conv_output = torch.matmul(X_unfolded, reshaped_kernels) + biases
        conv_output = conv_output.view(batch_size, out_height, out_width, out_channels)

        activation_images[:] = torch.relu(conv_output) + biases
        # print(f"Convolution Time: {time.time() - start_time:.4f} sec")
        return activation_images
        
        # return conv_output.view(batch_size, out_height, out_width, out_channels)


    def max_pooling_batch(self, stride, activation_images, max_pool_images):
        """
        Highly optimized max pooling using torch.nn.functional operations
        """
        # start_time = time.time()
        pool_size = 2
        # num_filters = activation_images.shape[3]

        # Reshape the activation images to (batch_size, num_filters, height, width)
        activation_images = activation_images.permute(0, 3, 1, 2)
        max_pool_images[:] = F.max_pool2d(activation_images, pool_size, stride=stride).permute(0, 2, 3, 1)
        # print(f"Max Pooling Time: {time.time() - start_time:.4f} sec")
        return max_pool_images

    def flatten(self, images):
        return images.view(images.size(0), -1)
    
    
    def padding(self, image, padding_size):
        batch_size = image.shape[0]
        padded_image = torch.zeros((batch_size, image.shape[1] + 2 * padding_size, image.shape[2] + 2 * padding_size, image.shape[3]))
        for f in range(batch_size):
            padded_image[f, padding_size:-padding_size, padding_size:-padding_size, :] = image[f]
        return padded_image
    
    def rotate_image(self, image, angle):
        return torch.rot90(image, 2)

    def backprop_convolution(self, d_kernel, d_bias, backprop_activation_images, patch_extraction_images):
        
        """
        Optimized gradient computation for the kernel and bias.
        - backprop_activation_images: Gradients flowing back from the upper layer.
        - patch_extraction_images: Input to the convolution operation from the forward pass.
        - d_kernel: Gradient of the kernels.
        - d_bias: Gradient of the biases.
        """
        batch_size, height, width, num_filters = backprop_activation_images.shape
        num_input_channels, kernel_height, kernel_width, _ = d_kernel.shape

        # Create patches for the convolution and gradient calculation
        patches = patch_extraction_images.unfold(1, kernel_height, 1).unfold(2, kernel_width, 1)

        # Compute gradients for the kernels
        for filter_index in range(num_filters):
            # Gradient for d_kernel: Perform batch matrix multiplication
            activations = backprop_activation_images[:, :, :, filter_index]  # (batch_size, 1, height, width)
            # Ensure activations have the correct shape for element-wise multiplication with patches
            activations = activations.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, height, width, 1, 1]

            # Gradients for the kernel are the sum of element-wise multiplication of activations and patches
            grad_kernel = torch.sum(activations.unsqueeze(-1) * patches, dim=(0, 1, 2)).permute(1, 2, 0)  # Sum across batch and spatial dimensions

            # Update d_kernel with the computed gradients
            d_kernel[filter_index] += grad_kernel

        # Compute gradients for the biases
        d_bias[0] += backprop_activation_images.sum(dim=(0, 1, 2))  # Sum over all dimensions except filter_index

        return d_kernel, d_bias
    
    def convolve_backprop_image_and_rotated_kernel(self, padded_activation_images, rotated_kernel, stride=1, filters_in_pooling_layer=None):
        num_filters, kernel_height, kernel_width, kernel_depth = rotated_kernel.shape
        batch_size, height, width, depth = padded_activation_images.shape

        # Ensure filters_in_pooling_layer is set correctly
        if filters_in_pooling_layer is None:
            filters_in_pooling_layer = num_filters

        # Calculate output dimensions
        output_height = (height - kernel_height) // stride + 1
        output_width = (width - kernel_width) // stride + 1

        # Reshape activation images to fit convolution operation
        padded_activation_images = padded_activation_images.permute(0, 3, 1, 2)  # (batch_size, depth, height, width)

        # Create a zero tensor to store the convolution result
        conv_result = torch.zeros((batch_size, output_height, output_width, filters_in_pooling_layer))
        conv_result = conv_result.permute(0, 3, 1, 2)  # (batch_size, num_filters, output_height, output_width)

        # Reshape the rotated kernel to match the dimensions of the activation images
        rotated_kernel = rotated_kernel.permute(3, 1, 2, 0)  # (kernel_depth, kernel_height, kernel_width, num_filters)
        reshaped_kernel = rotated_kernel.permute(0, 3, 1, 2)  # (kernel_depth, num_filters, kernel_height, kernel_width)

        for batch in range(batch_size):
            for filters in range(filters_in_pooling_layer):
                unfolded_images = F.unfold(padded_activation_images[batch], kernel_size=(kernel_height, kernel_width), stride=stride)
                result = torch.matmul(unfolded_images.permute(1, 0), reshaped_kernel[filters].reshape(-1, 1))   
                result = result.reshape(output_height, output_width)
                conv_result[batch, filters] = result

        
        conv_result = conv_result.permute(0, 2, 3, 1)  # (batch_size, output_height, output_width, num_filters)
        return conv_result
    
    def backprop_relu(self, activation_images):
        return torch.where(activation_images > 0, 1, 0)

    def backprop_max_pooling(self, d_pooled, activation_images, pool_size=2):
        
        """
        Optimized backpropagation through max pooling using vectorized operations.
        """

        batch_size, pooled_height, pooled_width, num_filters = d_pooled.shape
        original_height = pooled_height * pool_size
        original_width = pooled_width * pool_size

        # Reshape the activation images for pooling regions
        activation_images_reshaped = activation_images.view(
            batch_size,
            pooled_height,
            pool_size,
            pooled_width,
            pool_size,
            num_filters
        )
        # print("Activation Images Reshaped: ",activation_images_reshaped.shape)
        
        # Flatten the pooling dimensions for max computation
        flat_activation_images = activation_images_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(
            batch_size,
            pooled_height,
            pooled_width,
            num_filters,
            pool_size * pool_size
        )
        # print("Flat Activation Images: ",flat_activation_images.shape)
        
        # Find max indices within each pooling region
        max_indices = flat_activation_images.argmax(dim=-1, keepdim=True)
        # print("Max Indices: ",max_indices.shape)


        # Create a mask for the max values
        mask = torch.zeros_like(flat_activation_images)
        # print("Mask: ",mask.shape)
        mask.scatter_(-1, max_indices, 1)
        # print("Mask: ",mask.shape)

        # Reshape the mask back to match the activation images
        mask = mask.view(
            batch_size,
            pooled_height,
            pooled_width,
            num_filters,
            pool_size,
            pool_size
        ).permute(0, 1, 4, 2, 5, 3).reshape_as(activation_images_reshaped)

        # Expand the gradient from pooled layer to match the mask shape
        d_pooled_expanded = d_pooled.view(
            batch_size,
            pooled_height,
            1,
            pooled_width,
            1,
            num_filters
        ).expand(-1, -1, pool_size, -1, pool_size, -1)

        # Apply the mask to distribute gradients back
        d_activation_layer = (mask * d_pooled_expanded).reshape(
            batch_size,
            original_height,
            original_width,
            num_filters
        )

        return d_activation_layer

    def forward_pass_batch(self, X_batch):
        for i in range(self.conv_layers.size()):
            conv = self.conv_layers.getLayer(i)
            if i == 0:
                self.convolve_batch(X_batch, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            else:
                self.convolve_batch(self.conv_layers.getLayer(i - 1).max_pool_images, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            self.max_pooling_batch(2, conv.activation_images, conv.max_pool_images)

        batch_size = X_batch.shape[0]
        self.flattened_images = self.last_conv_layer.max_pool_images.view(batch_size, -1)
        self.final_input = torch.mm(self.flattened_images, self.fully_connected_weights) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def forward_pass_batch_conv2d(self, X_batch):
        for i in range(self.conv_layers.size()):
            conv = self.conv_layers.getLayer(i)
            if i == 0:
                self.convolve_batch_conv2d(X_batch, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            else:
                self.convolve_batch_conv2d(self.conv_layers.getLayer(i - 1).max_pool_images, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            self.max_pooling_batch(2, conv.activation_images, conv.max_pool_images)

        batch_size = X_batch.shape[0]
        self.flattened_images = self.last_conv_layer.max_pool_images.view(batch_size, -1)
        self.final_input = torch.mm(self.flattened_images, self.fully_connected_weights) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def backpropagation_batch(self, X, y, learning_rate):
        output_error = self.final_output - y
        output_derivative = self.sigmoid_derivative(self.final_output)
        d_output_delta = output_error * output_derivative

        self.fully_connected_weights -= learning_rate * torch.mm(self.flattened_images.T, d_output_delta)
        self.bias_output -= learning_rate * torch.sum(d_output_delta, dim=0, keepdim=True)

        d_flattened_image_delta = torch.mm(d_output_delta, self.fully_connected_weights.T)
        delta_loss_over_delta_max_pool_images = d_flattened_image_delta.view(self.last_conv_layer.max_pool_images.shape)

        for i in range(self.conv_layers.size() - 1, -1, -1):
            conv = self.conv_layers.getLayer(i)

            backprop_derivative_images = self.backprop_max_pooling(delta_loss_over_delta_max_pool_images, conv.activation_images)
            conv_relu_derivative = self.backprop_relu(conv.activation_images)

            backprop_activation_images = torch.multiply(backprop_derivative_images, conv_relu_derivative)

            d_kernel = torch.zeros_like(conv.kernels)

            d_bias = torch.zeros_like(self.bias_conv_layers[i])

            if i == 0:
                d_kernel, d_bias = self.backprop_convolution(d_kernel, d_bias, backprop_activation_images, X)
            else:
                d_kernel, d_bias = self.backprop_convolution(d_kernel, d_bias, backprop_activation_images, self.conv_layers.getLayer(i - 1).max_pool_images)

            conv.kernels -= learning_rate * d_kernel
            self.bias_conv_layers[i] -= learning_rate * d_bias

            if i == 0:
                continue
            else:
                rotated_kernel = torch.rot90(conv.kernels, 2)
                # print("Rotated Kernel: ",rotated_kernel.shape)
                # print("Backprop Activation Images: ",backprop_activation_images.shape)
                padded_backprop_images = self.padding(backprop_activation_images, conv.kernels.shape[1] - 1)
                # print("Padded Backprop Images: ",padded_backprop_images.shape)
                backprop_pool_images = self.convolve_backprop_image_and_rotated_kernel(padded_backprop_images, rotated_kernel, 1, self.conv_layers.getLayer(i - 1).max_pool_images.shape[3])
                delta_loss_over_delta_max_pool_images = backprop_pool_images

    def train(self, X, y, epochs, learning_rate, batch_size):
        """
        Train the neural network using balanced batch sampling and improved loss tracking.
        
        :param X: The images data.
        :param y: The labels for the images.
        :param epochs: The number of epochs to train the network.
        :param learning_rate: The learning rate for the network.
        :param batch_size: The number of images to use in each batch.
        """
        # Separate cat and dog indices
        cat_indices = torch.where(y[:, 0] == 1)[0]
        dog_indices = torch.where(y[:, 0] == 0)[0]
        
        # Get balanced dataset
        X_cat, y_cat = X[cat_indices], y[cat_indices]
        X_dog, y_dog = X[dog_indices], y[dog_indices]
        min_class_size = min(len(cat_indices), len(dog_indices))
        
        # Trim to balanced size
        X_cat, y_cat = X_cat[:min_class_size], y_cat[:min_class_size]
        X_dog, y_dog = X_dog[:min_class_size], y_dog[:min_class_size]
        
        # Batch size handling
        half_batch_size = batch_size // 2
        
        # Training loop with improved loss tracking
        for epoch in range(epochs):
            # Epoch-level metrics
            epoch_loss = 0.0
            epoch_correct_predictions = 0
            total_samples = 0
            
            # Shuffle indices for each class
            cat_indices = torch.randperm(len(X_cat))
            dog_indices = torch.randperm(len(X_dog))
            epoch_start_time = time.time()
            
            # Batch processing
            for i in range(0, min_class_size, half_batch_size):
                cat_batch_indices = cat_indices[i:i+half_batch_size]
                dog_batch_indices = dog_indices[i:i+half_batch_size]
                
                X_batch = torch.concatenate((
                    X_cat[cat_batch_indices], 
                    X_dog[dog_batch_indices]
                ), axis=0)
                
                # Shuffle batch
                batch_shuffle_indices = torch.randperm(len(X_batch))
                X_batch = X_batch[batch_shuffle_indices].to(X.device)
                # Skip if batch is incomplete
                if len(X_batch) < batch_size:
                    continue

                y_batch = torch.concatenate((
                    y_cat[cat_batch_indices], 
                    y_dog[dog_batch_indices]
                ), axis=0)
                y_batch = y_batch[batch_shuffle_indices].to(y.device)
                
                # Forward pass
                start_time = time.time()
                output = self.forward_pass_batch_conv2d(X_batch)
                
                # Loss calculation with numerical stability
                batch_loss = self.cross_entropy(y_batch, output)
                epoch_loss += batch_loss.item()
                
                # Backpropagation
                self.backpropagation_batch(X_batch, y_batch, learning_rate)
                
                # Prediction accuracy
                predictions = torch.argmax(output, dim=1)
                true_labels = torch.argmax(y_batch, dim=1)
                epoch_correct_predictions += (predictions == true_labels).sum().item()
                total_samples += len(X_batch)
                
                # print(f"Batch Time: {time.time() - start_time:.4f} sec")
            
            # Epoch-level metrics
            avg_epoch_loss = epoch_loss / (min_class_size // half_batch_size)
            accuracy = epoch_correct_predictions / total_samples * 100 if total_samples > 0 else 0
            
            # Logging and early stopping
            print(f"Epoch {epoch}:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Epoch Time: {time.time() - epoch_start_time:.4f} sec")
            # print(f"Latest Output: \n{output} \n Output Shape: {output.shape}")
            # print(f"Latest Target: \n{y_batch} \n Target Shape: {y_batch.shape}")
            print("--------------------------------------------------")
            print()
            
            # Early stopping conditions
            if avg_epoch_loss < 0.05 or math.isnan(avg_epoch_loss):
                print("Training converged or encountered instability.")
                break
            
            # Learning rate decay
            # Exponential decay formula
            learning_rate *= 0.9999
            # initial_learning_rate = 0.001
            # decay_rate = 0.999  # Decay multiplier per step
            # decay_steps = 16  # Number of epochs between decay

            # learning_rate = initial_learning_rate * (decay_rate ** (epoch / decay_steps))
            # print(f"Learning Rate: {learning_rate}")
        
        print("Training Completed.")

    def save(self, filename):
        """Save the trained network's weights and biases to a file."""

        torch.save({
            'input_nodes': self.input_nodes,
            'output_nodes': self.output_nodes,
            'fully_connected_weights': self.fully_connected_weights,
            'bias_output': self.bias_output,
            'bias_conv_layers': self.bias_conv_layers,
            'conv_layers': self.conv_layers,
        }, filename)

    def predict(self, X, batch_size):
        """
        Make predictions for the input data.
        :param X: The input data.
        :param batch_size: The batch size to use for making predictions.
        :return: The predictions.
        """
        predictions = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            if X_batch.shape[0] < batch_size:
                continue
            print("X Batch Shape: ",X_batch.shape)
            output = self.forward_pass_batch_conv2d(X_batch)
            predictions.append(output)
        return torch.cat(predictions)








# ##########################################
# # Convolutional Neural Network Parameters #
# ##########################################
# input_nodes = images.shape[1] * images.shape[2] * images.shape[3] # Number of input nodes equals the number of pixels in an image
# print("Images Shape: ",images.shape)
# output_nodes = 2
# target = labels
# print("Target Shape: ",target.shape)
# batch_size = 4
# # conv_layers = Convolutional_Layers_Torch(3, [[16,5], [32,3], [64,3]], ['relu', 'relu', 'relu'], images.shape, batch_size)
# conv_layers = Convolutional_Layers_Torch(2, [[16,5], [32,3]], ['relu', 'relu'], images.shape, batch_size)
# for i in range(len(conv_layers.conv_layers)):
#     print("Layer: ",i)
#     print("Kernels: ",conv_layers.conv_layers[i].kernels.shape)
#     print("Activation Images: ",conv_layers.conv_layers[i].activation_images.shape)
#     print("Max Pool Images: ",conv_layers.conv_layers[i].max_pool_images.shape)
#     print("Activation Function: ",conv_layers.conv_layers[i].activation_func)
#     print("\n")

# print("\n")
# cnn = Convolutional_Neural_Network(
#     input_nodes, 
#     output_nodes,
#     conv_layers,
#     )

# # cnn_2 = Convolutional_Neural_Network(
# #     input_nodes, 
# #     output_nodes,
# #     conv_layers,
# #     )

# start_time = time.time()
# output = cnn.forward_pass_batch_conv2d(images[:batch_size])
# print(f"Time Taken Forward Pass: {time.time() - start_time}")

# # start_time = time.time()
# # output_2 = cnn.forward_pass_batch_conv2d(images[:batch_size])
# # print(f"Time Taken Forward Pass Conv2d: {time.time() - start_time}")
# # print("\n")


# # start_time = time.time()
# # cnn.backpropagation_batch(images[:batch_size], target[:batch_size], 0.001)
# # print(f"Time Taken BackProp: {time.time() - start_time}")

# images = torch.cat((images[:30], images[1000:1030]), dim=0)
# target = torch.cat((target[:30], target[1000:1030]), dim=0)

# cnn.train(images, target, 10000, 0.001, batch_size)

# print("Activation Images: ",conv_layers.conv_layers[0].activation_images)
# Print images via cv2 for activation images layer 1
# for filter_index in range(cnn.conv_layers[0].activation_images.shape[3]):
#     activation_image = cnn.conv_layers.getLayer(0).activation_images[0, :, :, filter_index]
#     activation_image_2 = cnn_2.conv_layers.getLayer(0).activation_images[0, :, :, filter_index]
#     # print("Activation Image: ",activation_image)
#     activation_image = activation_image.detach().numpy()
#     activation_image_2 = activation_image_2.detach().numpy()
#     # print("Activation Image: ",activation_image)
#     print("Activation Image Shape: ",activation_image.shape)
#     print("Activation Image 2 Shape: ",activation_image_2.shape)
#     cv2.imshow(f"Activation Image {filter_index} Layer 1", activation_image)
#     cv2.imshow(f"CNN 2 Activation Image {filter_index} Layer 1", activation_image_2)
#     cv2.waitKey(0)


# # Print images via cv2 for pooling images layer 1
# for filter_index in range(cnn.conv_layers[0].max_pool_images.shape[3]):
#     max_pool_image = cnn.conv_layers.getLayer(0).max_pool_images[0, :, :, filter_index]
#     # print("Activation Image: ",activation_image)
#     max_pool_image = max_pool_image.detach().numpy()
#     # print("Activation Image: ",activation_image)
#     print("Max Pool Image Shape: ",max_pool_image.shape)
#     cv2.imshow(f"Max Pool Image {filter_index} Layer 1", max_pool_image)
#     cv2.waitKey(0)


# start_time = time.time()
# nn.backpropagation_batch(images[:batch_size], target[:batch_size], 0.001)
# print(f"Total time Taken BackProp: {time.time() - start_time}")
# images = torch.concatenate((images[:100], images[1000:1100]), axis=0)
# target = torch.concatenate((target[:100], target[1000:1100]), axis=0)
# images = torch.concatenate((images[:6], images[1000:1006]), axis=0)
# target = torch.cat((target[:6], target[1000:1006]), dim=0)
# print("Torch Images Shape: ",images.shape)
# print("Target Shape: ",target.shape)
# cnn.train(images, target, 1000, 0.01, batch_size)




# # Load the images
# cat_directory = os.path.join(BASE_DIR, 'validation_data', 'cat')
# cat_directory_files = os.listdir(cat_directory)

# images = []
# labels = []
# image_size = (64, 64)
# # Load 1000 cat images
# for filename in cat_directory_files:
#     img_path = os.path.join(cat_directory, filename)
#     assert os.path.exists(img_path), f"File not found: {img_path}"
#     try:
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, image_size)
#         images.append(img)
#         labels.append([1, 0])
#     except Exception as e:
#         # print(f"Error processing image {img_path}: {e}")
#         continue  # Skip corrupt images
# print(f"Loaded {len(images)} cat images.")
# images = np.array(images) / 255.0
# labels = np.array(labels)
# images = torch.tensor(images, dtype=torch.float32)
# labels = torch.tensor(labels, dtype=torch.float32)
# # Perform a forward pass
# predictions = cnn.predict(images, batch_size)
# print(predictions)

# cnn.save("cnn_model.pth")

# # Generate example matrix and kernel
# example_matrix = torch.randn(size=(10, 3, 64, 64))
# example_kernel = torch.randn(size=(2, 3, 3, 3))
# kernel_size = example_kernel.shape[2]

# # Print original shapes
# print("Example Matrix Shape: \n", example_matrix.shape)
# print("Example Matrix: \n", example_matrix)

# # Unfold the matrix
# unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=1)
# unfolded = unfold(example_matrix)
# print("\nUnfolded Shape: \n", unfolded.shape)
# print("Unfolded Matrix: \n", unfolded)

# # Print kernel shape
# print("\nExample Kernel Shape: \n", example_kernel.shape)
# print("\nExample Kernel: \n", example_kernel)

# # Reshape kernel and unfolded matrix for batch matrix multiplication
# # Reshape kernel to (out_channels, in_channels * kernel_height * kernel_width)
# kernel_bmm = example_kernel.view(example_kernel.size(0), -1)
# print("\nKernel BMM Shape: \n", kernel_bmm.shape)
# print("Kernel BMM: \n", kernel_bmm)

# # Reshape unfolded matrix to (batch_size, in_channels * kernel_height * kernel_width, num_patches)
# unfolded_bmm = unfolded.view(example_matrix.size(0), example_matrix.size(1) * kernel_size * kernel_size, -1)
# print("Unfolded BMM Shape: \n", unfolded_bmm.shape)
# print("Unfolded BMM: \n", unfolded_bmm)

# # Reshape kernel
# kernel_bmm = example_kernel.view(example_kernel.size(0), -1)
# print("\nKernel BMM Shape: \n", kernel_bmm.shape)

# # Reshape unfolded matrix to (batch_size, num_patches, in_channels * kernel_height * kernel_width)
# unfolded_bmm = unfolded.permute(0, 2, 1)
# print("\nUnfolded BMM Shape: \n", unfolded_bmm.shape)

# # Vectorized convolution
# # Compute dot product for all batches simultaneously
# conv_output = torch.matmul(unfolded_bmm, kernel_bmm.t())
# print("\nConv Output Shape: \n", conv_output.shape)

# # # Perform batch matrix multiplication
# # # This will compute the dot product for each batch and output channel
# # print("\nKernel Unsqueeze: \n", kernel_bmm.unsqueeze(0))
# # print("Kernel Unsqueeze Shape: \n", kernel_bmm.unsqueeze(0).shape)
# # conv_output = torch.bmm(kernel_bmm.unsqueeze(0), unfolded).squeeze(0)
# # print("Conv Output Shape: \n", conv_output.shape)

# # Reshape the output to typical convolution output dimensions
# output_height = example_matrix.size(2) - kernel_size + 1
# output_width = example_matrix.size(3) - kernel_size + 1
# reshaped_output = conv_output.view(example_matrix.size(0), example_kernel.size(0), output_height, output_width)
# print("Reshaped Output Shape: \n", reshaped_output.shape)
# print("Reshaped Output: \n", reshaped_output)



# relu_output = torch.relu(reshaped_output)
# print("ReLU Output Shape: \n", relu_output.shape)
# print("ReLU Output: \n", relu_output)











# unfolded_padded_activation_images = unfold(padded_activation_images)
            # print("Unfolded Padded Activation Images: ",unfolded_padded_activation_images.shape)
            # # Multiply the unfolded patches by the rotated kernel for each filter
            # kernel = rotated_kernel[k].view(1, depth * kernel_height * kernel_width, 1)  # Reshape rotated kernel for batch multiplication
            # print("Kernel: ",kernel.shape)
            # print("Unfolded Patches: ",unfolded_patches.view(batch_size, output_height * output_width, depth * kernel_height * kernel_width).shape)
            
            # # Compute the result of the convolution by batch matrix multiplication
            # result = torch.bmm(unfolded_patches.view(batch_size, output_height * output_width, depth * kernel_height * kernel_width), kernel) 

            # # Reshape the result to match the output dimensions
            # result = result.view(batch_size, output_height, output_width)
            
            # # Store the result in the backpropagated image
            # backprop_pooled_images[:, :, :, k] = result



# # Iterate over the number of filters
        # batch_size = backprop_activation_images.shape[0]
        # for f in range(batch_size):
        #     for filter_index in range(d_kernel.shape[0]):
        #         # Slide over the spatial dimensions of backprop_activation_images
        #         for i in range(backprop_activation_images.shape[1]):
        #             for j in range(backprop_activation_images.shape[2]):
        #                 # Define the patch of the max pool images corresponding to this position
        #                 start_i = i  # Adjust for stride if necessary
        #                 end_i = start_i + d_kernel.shape[1]
        #                 start_j = j  # Adjust for stride if necessary
        #                 end_j = start_j + d_kernel.shape[2]

        #                 # Extract the patch
        #                 patch = patch_extraction_images[f, start_i:end_i, start_j:end_j, :d_kernel.shape[3]]

        #                 # Accumulate gradient for the kernel
        #                 d_kernel[filter_index] += backprop_activation_images[f, i, j, filter_index] * patch

        #         # Accumulate gradient for the bias
        #         d_bias[0, filter_index] += torch.sum(backprop_activation_images[f, :, :, filter_index])

        # return d_kernel, d_bias
        # 
        # 
        # 
        # 
        # # """
        # Backpropagate through the pooling layer.
        # :param d_pooled: Gradient from the upper layer.
        # :param activation_images: Activations before pooling.
        # """
        # batch_size, height, width, num_filters = d_pooled.shape
        # d_activation_layer = torch.zeros_like(activation_images)
        # for f in range(batch_size):
        #     for i in range(num_filters):
        #         for j in range(height):
        #             for k in range(width):
        #                 # Get the pooling region in the activation image
        #                 h_start, h_end = j * pool_size, (j + 1) * pool_size
        #                 w_start, w_end = k * pool_size, (k + 1) * pool_size
                        
        #                 # Get pooling region where the max value occurred
        #                 pooling_region = activation_images[f, h_start:h_end, w_start:w_end, i]

        #                 # Get the max value location
        #                 max_val = torch.max(pooling_region)
        #                 mask = pooling_region == max_val            

        #                 # Update the new activation layer
        #                 d_activation_layer[f, h_start:h_end, w_start:w_end, i] = d_pooled[f, j, k, i] * mask

        # return d_activation_layer