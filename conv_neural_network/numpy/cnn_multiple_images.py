import math
import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
from PIL import UnidentifiedImageError, Image
import warnings
import time
from conv_neural_network.numpy.load_cnn_training_data_np import images, labels

warnings.filterwarnings("ignore", message="Corrupt JPEG data")
BASE_DIR = os.path.dirname(os.path.abspath(__name__))



class Convulutional_Neural_Network():

    def __init__(self, input_nodes, output_nodes, conv_layer_kernels, conv_layer_activation_images, conv_layer_max_pool_images, conv_layer_2_activation_images, conv_layer_2_kernels, conv_layer_2_max_pool_images, fully_connected_weights=None, bias_output=None, bias_conv_layer=None, bias_conv_layer_2=None):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        print("Input Nodes: ",self.input_nodes)
        print("Output Nodes: ",self.output_nodes)
        self.conv_layer_kernels = conv_layer_kernels
        self.conv_layer_activation_images = conv_layer_activation_images
        self.conv_layer_max_pool_images = conv_layer_max_pool_images
        self.conv_layer_2_activation_images = conv_layer_2_activation_images
        self.conv_layer_2_kernels = conv_layer_2_kernels
        self.conv_layer_2_max_pool_images = conv_layer_2_max_pool_images

        # Intialize weights and bias for the fully connected layer
        if fully_connected_weights is not None:
            self.fully_connected_weights = fully_connected_weights
        else:
            self.fully_connected_weights = np.random.randn(self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2] * self.conv_layer_2_max_pool_images.shape[3], self.output_nodes) * np.sqrt(2 / (self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2] * self.conv_layer_2_max_pool_images.shape[3] + self.output_nodes))
        print("Full Connected Weights Shape: ",self.fully_connected_weights.shape)
        if bias_output is not None:
            self.bias_output = bias_output
        else:
            self.bias_output = np.random.randn(1, self.output_nodes) * np.sqrt(2 / (self.conv_layer_2_max_pool_images.shape[0] * self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2] + self.output_nodes))
        print("Output Bias Shape: ",self.bias_output.shape)


        # Intialize bias for the convolutional layer
        if bias_conv_layer is not None:
            self.bias_conv_layer = bias_conv_layer
        else:
            self.bias_conv_layer = np.random.randn(1, self.conv_layer_kernels.shape[0]) * 0.01
        
        if bias_conv_layer_2 is not None:
            self.bias_conv_layer_2 = bias_conv_layer_2
        else:
            self.bias_conv_layer_2 = np.random.randn(1, self.conv_layer_2_kernels.shape[0]) * 0.01
        print("Conv Layer Bias Shape:\n ",self.bias_conv_layer.shape)
        print("Conv Layer 2 Bias Shape:\n ",self.bias_conv_layer_2.shape)
    
    def initialize_weights(input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(2 / (input_dim + output_dim))

    
    def relu(self, x):
        return np.maximum(0, x)    
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
    
    def sigmoid_derivative(self, output):
        return output * (1 - output)
    
    def mean_squared_error(self, target_output, output):
        return np.mean(np.square(target_output - output))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
    def softmax_derivative(self, output):
        n = len(output)
        jacobian = output.reshape(-1, 1) * np.eye(n) - np.dot(output, output.T)
        return jacobian
    
    def cross_entropy(self, target_output, output):
        return -np.sum(target_output * np.log(output))
    
    def calculate_delta(self, error, derivative):
        return error * derivative
    
    def calculate_loss(self, y_true, y_pred):
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0) issues
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def calculate_error(self, target_output, final_output):
        return final_output - target_output
    
    def convulate_batch(self, X_batch, kernels, biases, activation_images, activation_func, stride):
        """
        Convolves each image in the batch `X_batch` with the given `kernels` and `biases`,
        and stores the result in `activation_images` after applying the specified `activation_func`.
        
        X_batch: Input batch of images (batch_size, height, width, channels)
        kernels: Convolutional kernels (num_kernels, kernel_height, kernel_width, num_input_channels)
        biases: Bias terms for the convolutional layer (num_kernels,)
        activation_images: Output activation images (num_kernels, height, width)
        activation_func: Activation function ('relu', etc.)
        stride: Stride used in the convolution operation
        """
        
        batch_size, height, width, channels = X_batch.shape
        num_kernels, kernel_height, kernel_width, _ = kernels.shape
        # print("Kernel Shape: ",kernels.shape)
        
        # Apply convolution to each image in the batch
        for f in range(batch_size):
            img = X_batch[f]
            for i in range(num_kernels):
                for j in range(activation_images.shape[1]):
                    for k in range(activation_images.shape[2]):
                        # Calculate the starting and ending index of the patch for this filter
                        start_row = j * stride
                        end_row = start_row + kernels.shape[1]
                        start_col = k * stride
                        end_col = start_col + kernels.shape[2]

                        patch = img[start_row:end_row, start_col:end_col, :kernels.shape[3]]
                        # print("Patch Shape: ",patch.shape)
                        # print("Kernel Shape: ",kernels[i].shape)
                
                        conv_value = np.sum(patch * kernels[i]) + biases[0, i]

                        # Apply the ReLU activation function
                        if activation_func == 'relu':
                            relu_value = self.relu(conv_value)
                        else:
                            relu_value = self.sigmoid(conv_value)
        
                        activation_images[f, j, k, i] = relu_value
        
        return activation_images
    
    def max_pooling_batch(self, stride, activation_images, max_pool_images):
        """
        Perform max pooling on a batch of activation images.
        :param stride: The stride for the max pooling operation.
        :param activation_images: The activation images to pool.
        :param max_pool_images: The output max-pooled images.
        """
        print()
        pool_size = 2
        num_filters = activation_images.shape[3]
        for f in range(activation_images.shape[0]):
            image = activation_images[f]
            for i in range(num_filters):
                for j in range(0, activation_images.shape[1] - pool_size + 1, stride):
                    for k in range(0, activation_images.shape[2] - pool_size + 1, stride):
                        patch = image[j:j+pool_size, k:k+pool_size, i]
                        max_value = np.max(patch)
                        max_pool_images[f, j // stride, k // stride, i] = max_value

        return max_pool_images
    
    
    def flatten(self, images):
        flattened_images = [image.flatten() for image in images]
        flattened_images = np.array(flattened_images)
        flattened_images = flattened_images.flatten()
        return flattened_images
    
    def padding(self, image, padding_size):
        batch_size = image.shape[0]
        padded_image = np.zeros((batch_size, image.shape[1] + 2 * padding_size, image.shape[2] + 2 * padding_size, image.shape[3]))
        for f in range(batch_size):
            padded_image[f, padding_size:-padding_size, padding_size:-padding_size, :] = image[f]
        return padded_image
    
    def rotate_np_image(self, image, angle):
        return np.rot90(image, 2)

    def backprop_convolution(self, d_kernel, d_bias, backprop_activation_images, patch_extraction_images):

        # Iterate over the number of filters
        batch_size = backprop_activation_images.shape[0]
        for f in range(batch_size):
            for filter_index in range(d_kernel.shape[0]):
                # Slide over the spatial dimensions of backprop_activation_images
                for i in range(backprop_activation_images.shape[1]):
                    for j in range(backprop_activation_images.shape[2]):
                        # Define the patch of the max pool images corresponding to this position
                        start_i = i  # Adjust for stride if necessary
                        end_i = start_i + d_kernel.shape[1]
                        start_j = j  # Adjust for stride if necessary
                        end_j = start_j + d_kernel.shape[2]

                        # Extract the patch
                        patch = patch_extraction_images[f, start_i:end_i, start_j:end_j, :d_kernel.shape[3]]

                        # Accumulate gradient for the kernel
                        d_kernel[filter_index] += backprop_activation_images[f, i, j, filter_index] * patch

                # Accumulate gradient for the bias
                d_bias[0, filter_index] += np.sum(backprop_activation_images[f, :, :, filter_index])

        return d_kernel, d_bias
    
    def convolve_backprop_image_and_rotated_kernel(self, padded_activation_images, rotated_kernel, stride=1):
        """
        Perform convolution between the padded activation images and the rotated kernel.
        :param padded_activation_images: Padded backpropagated activation images of shape (height, width, depth).
        :param rotated_kernel: The kernel rotated by 180 degrees.
        :param stride: The stride for the convolution. Default is 1.
        :return: Convolution result (gradient with respect to activations).
        """
        # Unpack the kernel dimensions
        num_filters, kernel_height, kernel_width, kernel_depth = rotated_kernel.shape
        
        # Get the dimensions of the padded activation images
        batch_size, height, width, depth = padded_activation_images.shape
        
        # Calculate the output dimensions
        output_height = (height - kernel_height) // stride + 1
        output_width = (width - kernel_width) // stride + 1
        
        # Initialize the gradient (this will be the result of the convolution)
        backprop_pooled_images = np.zeros((batch_size, output_height, output_width, self.conv_layer_max_pool_images.shape[3]))
        # print("Backprop Pooled Images Shape: ",backprop_pooled_images.shape)
        
        # Perform the convolution
        for f in range(batch_size):
            for k in range(num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract the patch from the padded activation images
                        patch = padded_activation_images[f, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width, :kernel_depth]
                        
                        # For each filter, convolve the patch with the rotated kernel
                        for k in range(self.conv_layer_max_pool_images.shape[3]):  # Loop over filters in the kernel
                            backprop_pooled_images[f, i, j, k] = np.sum(patch * rotated_kernel[k])  # Sum over depth and kernel

        return backprop_pooled_images
    
    def backprop_relu(self, d_relu, activation_images):
        """
        Backpropagate through the ReLU activation function.
        :param d_relu: Gradient from the upper layer.
        :param activation_images: Activations before the ReLU function.
        """
        d_activation = np.where(activation_images > 0, d_relu, 0)
        return d_activation
    
    def backprop_calculate_relu_derivative(self, activation_images):
        """
        Calculate the derivative of the ReLU activation function.
        :param activation_images: Activations before the ReLU function.
        """
        return np.where(activation_images > 0, 1, 0)

    def backprop_max_pooling(self, d_pooled, activation_images, pool_size=2):
        """
        Backpropagate through the pooling layer.
        :param d_pooled: Gradient from the upper layer.
        :param activation_images: Activations before pooling.
        """
        batch_size, height, width, num_filters = d_pooled.shape
        d_activation_layer = np.zeros_like(activation_images)
        for f in range(batch_size):
            for i in range(num_filters):
                for j in range(height):
                    for k in range(width):
                        # Get the pooling region in the activation image
                        h_start, h_end = j * pool_size, (j + 1) * pool_size
                        w_start, w_end = k * pool_size, (k + 1) * pool_size
                        
                        # Get pooling region where the max value occurred
                        pooling_region = activation_images[f, h_start:h_end, w_start:w_end, i]

                        # Get the max value location
                        max_val = np.max(pooling_region)
                        mask = pooling_region == max_val            

                        # Update the new activation layer
                        d_activation_layer[f, h_start:h_end, w_start:w_end, i] = d_pooled[f, j, k, i] * mask

        return d_activation_layer
    
    def forward_pass_batch(self, X_batch):
        """
        Forward pass for a batch
        """
        self.convulate_batch(X_batch, self.conv_layer_kernels, self.bias_conv_layer, self.conv_layer_activation_images, 'relu', 1)
        self.max_pooling_batch(2, self.conv_layer_activation_images, self.conv_layer_max_pool_images)
        self.convulate_batch(self.conv_layer_max_pool_images, self.conv_layer_2_kernels, self.bias_conv_layer_2, self.conv_layer_2_activation_images, 'relu', 1)
        self.max_pooling_batch(2, self.conv_layer_2_activation_images, self.conv_layer_2_max_pool_images)

        batch_size = X_batch.shape[0]
        # print("X Batch Shape: ",X_batch.shape)
        # print("Batch Size: ",batch_size)
        # print("Conv Layer 2 Max Pool Images Shape: ",self.conv_layer_2_max_pool_images.shape)
        self.flattened_images = self.conv_layer_2_max_pool_images.reshape(batch_size, -1)
        # print("Flattened Images Shape: ",self.flattened_images.shape)

        self.final_input = np.dot(self.flattened_images, self.fully_connected_weights) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output


    def backpropagation_batch(self, X, y, learning_rate):
        ############################################################################################################
        ############################################################################################################
        
        # Calculate the output error
        output_error = self.final_output - y  # For binary classification, this is the error (prediction - true)
        
        # Calculate the gradient of the output layer
        output_derivative = self.sigmoid_derivative(self.final_output)
        d_output_delta = output_error * output_derivative

        self.flattened_images = self.flattened_images.T
        
        # Update the weights using gradient descent (assuming no momentum, for simplicity)
        weights_before = self.fully_connected_weights.copy()
        self.fully_connected_weights -= learning_rate * np.dot(self.flattened_images, d_output_delta)

        self.bias_output -= learning_rate * np.sum(d_output_delta)

        # ############################################################################################################
        # ############################################################################################################
        d_flattened_image_delta = np.dot(d_output_delta, self.fully_connected_weights.T)
        delta_loss_over_delta_max_pool_2_images = d_flattened_image_delta.reshape(self.conv_layer_2_max_pool_images.shape)
        

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the second pooling layer

        # Calculate the gradient of the loss with respect to the relu activation
        backprop_derivative_images_layer_2 = self.backprop_max_pooling(delta_loss_over_delta_max_pool_2_images, self.conv_layer_2_activation_images)

        # Calculate the gradient of the loss with respect to the convultional layer 2 images
        conv_layer_2_relu_derivative = self.backprop_calculate_relu_derivative(self.conv_layer_2_activation_images)
        backprop_activation_images_layer_2 = np.multiply(backprop_derivative_images_layer_2, conv_layer_2_relu_derivative)
        # print("Backprop Activation Images Layer 2 Shape: ",backprop_activation_images_layer_2.shape)

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the second convolutional layer

        # Calculate the gradient of the loss with respect to the kernel and bias
        d_kernel_2 = np.zeros_like(self.conv_layer_2_kernels)
        # print("Kernel 2 Shape: ",d_kernel_2.shape)
        d_bias_2 = np.zeros_like(self.bias_conv_layer_2)
        # print("Bias 2 Shape: ",d_bias_2.shape)
        d_kernel_2, d_bias_2 = self.backprop_convolution(d_kernel_2, d_bias_2, backprop_activation_images_layer_2, self.conv_layer_max_pool_images)

        # Update the kernel and bias for the second convolutional layer
        kernel_before = self.conv_layer_2_kernels.copy()
        self.conv_layer_2_kernels -= learning_rate * d_kernel_2
        # print("Kernel Layer 2 Weights Change: ",np.sum(kernel_before - self.conv_layer_2_kernels))
        self.bias_conv_layer_2 -= learning_rate * d_bias_2

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the first pooling layer

        # Rotate the kernel by 180 degrees
        rotated_kernel = np.rot90(self.conv_layer_2_kernels, 2)
        # print("Rotated Kernel Shape: ",rotated_kernel.shape)

        # Add padding to the backpropagated activation images
        padding_size = self.conv_layer_2_kernels.shape[1] - 1
        padded_backpropagated_activation_images = self.padding(backprop_activation_images_layer_2, padding_size)

        # Convolve the padded activation images with the rotated kernel
        backprop_pool_2_images = self.convolve_backprop_image_and_rotated_kernel(padded_backpropagated_activation_images, rotated_kernel, 1)
        # print("Backprop Pool 2 Images Shape: ",backprop_pool_2_images.shape)

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the first convolutional layer

        # Calculate backpropagation through the ReLU activation function
        # Change backprop_pool_2_images to backprop_activation_images_layer_1
        backprop_derivative_activation_images_layer_1 = self.backprop_max_pooling(backprop_pool_2_images, self.conv_layer_activation_images)


        # Calculate the gradient of the loss with respect to the relu activation
        conv_layer_relu_derivative = self.backprop_calculate_relu_derivative(self.conv_layer_activation_images)

        # Calculate the gradient of the loss with respect to the activation images
        backprop_activation_images_layer_1 = np.multiply(backprop_derivative_activation_images_layer_1, conv_layer_relu_derivative)


        ############################################################################################################
        ############################################################################################################
        # Get the kernel and bias for the first convolutional layer
        d_kernel_1 = np.zeros_like(self.conv_layer_kernels)
        d_bias_1 = np.zeros_like(self.bias_conv_layer)
        d_kernel_1, d_bias_1 = self.backprop_convolution(d_kernel_1, d_bias_1, backprop_activation_images_layer_1, X)


        # Update the kernel and bias for the first convolutional layer
        kernel_layer_1_before = self.conv_layer_kernels.copy()
        self.conv_layer_kernels -= learning_rate * d_kernel_1
        # print("Kernel Layer 1 Weights change: ",np.sum(kernel_layer_1_before - self.conv_layer_kernels))
        self.bias_conv_layer -= learning_rate * d_bias_1
        print()

    def train(self, X, y, epochs, learning_rate, batch_size):
        """
        Train the neural network using the specified training data.
        :param X: The images data.
        :param y: The labels for the images.
        :param epochs: The number of epochs to train the network.
        :param learning_rate: The learning rate for the network.
        :param batch_size: The number of images to use in each batch.
        """
        # Create an array of indices for categorical data
        cat_indices = np.where(y[:, 0] == 1)[0] # Get the indices of the cat images
        dog_indices = np.where(y[:, 1] == 1)[0] # Get the indices of the dog images

        X_cat, y_cat = X[cat_indices], y[cat_indices]  # Get the cat images and labels
        X_dog, y_dog = X[dog_indices], y[dog_indices]  # Get the dog images and labels

        min_class_size = min(len(cat_indices), len(dog_indices))  # Get the size of the smallest class
        X_cat, y_cat = X_cat[:min_class_size], y_cat[:min_class_size]  # Get the first `min_class_size` cat images and labels
        X_dog, y_dog = X_dog[:min_class_size], y_dog[:min_class_size]  # Get the first `min_class_size` dog images and labels

        # Calculate batch size for each class
        half_batch_size = batch_size // 2

        print(f"Total cat images: {len(X_cat)}",f"Total dog images: {len(X_dog)}")
        loss = 0

        for epoch in range(epochs):
            cat_indices = np.random.permutation(len(X_cat))  # Shuffle the cat indices
            dog_indices = np.random.permutation(len(X_dog))  # Shuffle the dog indices
            # print("Cat Indices: ",len(cat_indices))
            for i in range(0, len(X_cat), half_batch_size):
                cat_batch_indices = cat_indices[i:i+half_batch_size]
                dog_batch_indices = dog_indices[i:i+half_batch_size]

                X_batch = np.concatenate((X_cat[cat_batch_indices], X_dog[dog_batch_indices]), axis=0)
                # print("X Batch Shape: ",X_batch.shape)
                y_batch = np.concatenate((y_cat[cat_batch_indices], y_dog[dog_batch_indices]), axis=0)
                # print("Y Batch Shape: ",y_batch.shape)
                batch_shuffle_indices = np.random.permutation(len(X_batch))

                X_batch = X_batch[batch_shuffle_indices]
                y_batch = y_batch[batch_shuffle_indices]

                current_batch_size = X_batch.shape[0]
                if current_batch_size < batch_size:
                    continue
                
                start_time = time.time()
                output = self.forward_pass_batch(X_batch)
                loss = self.cross_entropy(y_batch, output)

                if loss < 0.10 or math.isnan(loss):
                    print(f"Epoch {epoch}, Loss: {loss}")
                    print("Training Completed")
                    print("Exiting Training.....")
                    return
                
                self.backpropagation_batch(X_batch, y_batch, learning_rate)
                print(f"Epoch {epoch}, Loss: {loss}, Remaining Data: {len(X_cat) - (i + half_batch_size)}")
                print("Time Taken: ",time.time() - start_time)
                
       
                    
    def save(self, filename):
        """Save the trained network's weights and biases to a file."""
        # Save weights and biases
        np.savez(filename, 
                input_nodes=self.input_nodes,
                output_nodes=self.output_nodes,
                conv_layer_kernels=self.conv_layer_kernels,
                conv_layer_activation_images=self.conv_layer_activation_images,
                conv_layer_max_pool_images=self.conv_layer_max_pool_images,
                conv_layer_2_kernels=self.conv_layer_2_kernels,
                conv_layer_2_activation_images=self.conv_layer_2_activation_images,
                conv_layer_2_max_pool_images=self.conv_layer_2_max_pool_images,
                fully_connected_weights=self.fully_connected_weights,
                bias_output=self.bias_output,
                bias_conv_layer=self.bias_conv_layer,
                bias_conv_layer_2=self.bias_conv_layer_2)
        print(f"Model saved to {filename}")

    
    def predict(self, X_single):
        """
        Predict a single image using the trained CNN model.
        
        X_single: The single image to predict (height, width, channels)
        
        Returns: The predicted output for the image (a vector for classification tasks)
        """
        # Reshape the image into a batch of size 1
        # X_single = np.expand_dims(X_single, axis=0)  # Shape becomes (1, height, width, channels)
        print("X Single Shape: ",X_single.shape)

        new_batch_size = X_single.shape[0]
        self.conv_layer_activation_images = np.zeros((
            new_batch_size,
            self.conv_layer_activation_images.shape[1],
            self.conv_layer_activation_images.shape[2],
            self.conv_layer_activation_images.shape[3]))
        self.conv_layer_max_pool_images = np.zeros((
            new_batch_size,
            self.conv_layer_max_pool_images.shape[1],
            self.conv_layer_max_pool_images.shape[2],
            self.conv_layer_max_pool_images.shape[3]))
        self.conv_layer_2_activation_images = np.zeros((
            new_batch_size,
            self.conv_layer_2_activation_images.shape[1],
            self.conv_layer_2_activation_images.shape[2],
            self.conv_layer_2_activation_images.shape[3]))
        self.conv_layer_2_max_pool_images = np.zeros((
            new_batch_size,
            self.conv_layer_2_max_pool_images.shape[1],
            self.conv_layer_2_max_pool_images.shape[2],
            self.conv_layer_2_max_pool_images.shape[3]))
        

        
        # Perform the forward pass with a batch size of 1
        output = self.forward_pass_batch(X_single)
        
        return output
    

        
    
            


###########################################
# Convolutional Neural Network Parameters #
###########################################
input_nodes = images.shape[1] * images.shape[2] * images.shape[3] # Number of input nodes equals the number of pixels in an image
output_nodes = 2
target = labels
batch_size = 2




############################################
# Convolutional Layer 1 Parameters #
############################################
conv_layer_kernels = np.random.randn(8, 5, 5, images.shape[3]) * np.sqrt(2 / (5 * 5 * images.shape[3]))
conv_layer_activation_images = (
np.zeros((
    batch_size,
    images.shape[1] - conv_layer_kernels.shape[1] + 1,
    images.shape[2] - conv_layer_kernels.shape[2] + 1,
    conv_layer_kernels.shape[0]
)))   

conv_layer_max_pooling_images = (
np.zeros((
    batch_size,
    conv_layer_activation_images.shape[1] // 2,
    conv_layer_activation_images.shape[2] // 2,
    conv_layer_activation_images.shape[3],
)))


############################################
# Convolutional Layer 2 Parameters #
############################################
conv_layer_2_kernels = np.random.randn(16, 3, 3, conv_layer_max_pooling_images.shape[3]) * np.sqrt(2 / (3 * 3 * conv_layer_max_pooling_images.shape[3]))
conv_layer_2_activation_images = (
np.zeros((
    batch_size,
    conv_layer_max_pooling_images.shape[1] - conv_layer_2_kernels.shape[1] + 1,
    conv_layer_max_pooling_images.shape[2] - conv_layer_2_kernels.shape[2] + 1,
    conv_layer_2_kernels.shape[0]
)))

conv_layer_2_max_pooling_images = (
np.zeros((
    batch_size,
    conv_layer_2_activation_images.shape[1] // 2,
    conv_layer_2_activation_images.shape[2] // 2,
    conv_layer_2_activation_images.shape[3]
)))

############################################
############################################
print()                 

print("Conv Layer Kernels Shape: ",conv_layer_kernels.shape)
print("Activation Layer 1 Shape: \n",conv_layer_activation_images.shape)
print("Conv Layer 1 Max Pooling Layers Shape: ",conv_layer_max_pooling_images.shape)


print("\n")


print("Conv Layer 2 Activation Images Shape: \n",conv_layer_2_activation_images.shape)
print("Conv Layer 2 Kernels Shape: \n",conv_layer_2_kernels.shape)
print("Conv Layer 2 Max Pooling Layers Shape: \n",conv_layer_2_max_pooling_images.shape)

print("\n")
nn = Convulutional_Neural_Network(
    input_nodes, 
    output_nodes,
    conv_layer_kernels, 
    conv_layer_activation_images=conv_layer_activation_images, 
    conv_layer_max_pool_images=conv_layer_max_pooling_images,
    conv_layer_2_kernels=conv_layer_2_kernels,
    conv_layer_2_activation_images=conv_layer_2_activation_images,
    conv_layer_2_max_pool_images=conv_layer_2_max_pooling_images
    )


# nn.forward_pass_batch(images[:32])
# nn.backpropagation_batch(images[:32], target[:32], 0.01)
images = np.concatenate((images[:4], images[1000:1004]), axis=0)
print("Images Shape: ",images.shape)
target = np.concatenate((target[:4], target[1000:1004]), axis=0)
print("Target Shape: ",target.shape)
nn.train(images, target, 10000, 0.01, batch_size)
# nn.save('trained_cnn_model.npz')
# prediction_image = cv2.imread(os.path.join(BASE_DIR, 'training_data', 'cat', '10580.jpg'))
# prediction_image = cv2.resize(prediction_image, (64, 64))
# prediction_image = np.array([prediction_image]) / 255.0
# prediction = nn.predict(prediction_image)
# print("Prediction: ",prediction)

# output = nn.forward_pass_batch(images)
# nn.backpropagation(images, target, 0.01)

# act_imgs = nn.convulate_batch(images, nn.conv_layer_kernels, nn.bias_conv_layer, nn.conv_layer_activation_images, 'relu', 1)
# print("Activation Images Shape: ",act_imgs.shape)
# max_pool_images = nn.max_pooling_batch(2, act_imgs, nn.conv_layer_max_pool_images)
# print("Max Pool Images Shape: ",max_pool_images.shape)
# act_imgs_2 = nn.convulate_batch(max_pool_images, nn.conv_layer_2_kernels, nn.bias_conv_layer_2, nn.conv_layer_2_activation_images, 'relu', 1)
# print("Activation Images 2 Shape: ",act_imgs_2.shape)
# max_pool_images_2 = nn.max_pooling_batch(2, act_imgs_2, nn.conv_layer_2_max_pool_images)
# print("Max Pool Images 2 Shape: ",max_pool_images_2.shape)

# for batch in range(act_imgs.shape[0]):
#     for filter_index in range(act_imgs.shape[3]):
#         activation_image = act_imgs[batch, :, :, filter_index]
#         cv2.imshow(f"Batch {batch}, Activation Layer 1 Image {filter_index} : ",activation_image)
#         cv2.waitKey(0)

# for batch in range(max_pool_images.shape[0]):
#     for filter_index in range(max_pool_images.shape[3]):
#         max_pool_image = max_pool_images[batch, :, :, filter_index]
#         cv2.imshow(f"Batch {batch}, Max Pool Image {filter_index}", max_pool_image)
#         cv2.waitKey(0)

# for batch in range(act_imgs_2.shape[0]):
#     for filter_index in range(act_imgs_2.shape[3]):
#         activation_image = act_imgs_2[batch, :, :, filter_index]
#         cv2.imshow(f"Batch {batch}, Activation Layer 2 Image {filter_index}", activation_image)
#         cv2.waitKey(0)

# for batch in range(max_pool_images_2.shape[0]):
#     for filter_index in range(max_pool_images_2.shape[3]):
#         max_pool_image = max_pool_images_2[batch, :, :, filter_index]
#         cv2.imshow(f"Batch {batch}, Max Pool Image 2 {filter_index}", max_pool_image)
#         cv2.waitKey(0)


# # start_time = time.time()
# # final_output = nn.forward_pass(images)
# # backprop = nn.backpropagation(images, target, 0.01)
# # print("Time Taken For One Epoch: ",time.time() - start_time)
# nn.train(images, target, 10000, 0.5)


# nn.convulate(images, nn.conv_layer_kernels, 1)

# nn.max_pooling(2, nn.conv_layer_activation_images, nn.conv_layer_max_pool_images)

# nn.convulate_2(nn.conv_layer_max_pool_images, nn.conv_layer_2_kernels, 1)

# nn.max_pooling(2, nn.conv_layer_2_activation_images, nn.conv_layer_2_max_pool_images)





# for filter_index in range(nn.conv_layer_activation_images.shape[2]):
    
#     activation_image = conv_layer_activation_images[:, :, filter_index]

#     cv2.imshow(f"Activation Image {filter_index} Shape: ",activation_image)
#     cv2.waitKey(0)

# for filter_index in range(nn.conv_layer_activation_images.shape[2]):
#     max_pool_image = nn.conv_layer_max_pool_images[:, :, filter_index]
#     print("Max Pool Image Shape: ",max_pool_image.shape)
#     cv2.imshow(f"Max Pool Image {filter_index}", max_pool_image)
#     cv2.waitKey(0)

# for filter_index in range(nn.conv_layer_2_activation_images.shape[2]):
#     activation_layer_2_image = nn.conv_layer_2_activation_images[:, :, filter_index]
#     print(f"Activation Layer 2 Image {filter_index} Shape : ",activation_layer_2_image.shape)
#     cv2.imshow(f"Activation Layer 2 Image {filter_index}", activation_layer_2_image)
#     cv2.waitKey(0)

# # print("Conv Layer 2 Activation Images Shape: ",nn.conv_layer_2_activation_images.shape)
# for filter_index in range(nn.conv_layer_2_activation_images.shape[2]):
#     max_pool_image = nn.conv_layer_2_max_pool_images[:, :, filter_index]
#     print("Max Pool 2 Image Shape: ",max_pool_image.shape)
#     cv2.imshow(f"Max Pool Layer 2 Image {filter_index}", max_pool_image)
#     cv2.waitKey(0)

# print("Activation Layer 1 Size: ",len(nn.conv_layer_activation_images))
# Scale the values to 0-255 if they are in the range 0-1 (assuming ReLU or similar)
    # activation_image = np.clip(activation_image * 255, 0, 255).astype(np.uint8)
    # cv2.imshow(f"Filter {filter_index + 1}, Activation Image {filter_index}", nn.conv_layer_activation_images[:, :, filter_index])
    # Extract the activation image for the current filter


# nn.train(images, target, 10000, 0.01)



# print("Conv Layer Activation Layers: \n",nn.conv_layer_activation_images)
# print("Conv Layer Activation Layers Shape: ",nn.conv_layer_activation_images.shape) 

# Train the neural ns















# Load the training data for cats


# # Load the training data for dogs
# dog_directory = os.path.join(BASE_DIR, 'training_data', 'dog')
# dog_directory_files = os.listdir(dog_directory)
# for filename in dog_directory_files:
#     img_path = os.path.join(dog_directory, filename)
#     assert os.path.exists(img_path), f"File not found: {img_path}"
#     try:
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, image_size)
#         images.append(img)
#         labels.append(0)
#     except Exception as e:
#         # print(f"Error processing image {img_path}: {e}")
#         continue  # Skip corrupt



# nn.convulate(images, nn.conv_layer_kernels, 1)

# print("Conv Layer Activation Shape: ",nn.conv_layer_activation_images[0].shape)
# print()

# print("Conv Layer 2 Kernels Shape: ",nn.conv_layer_2_kernels.shape)
# # print("Conv Layer 2 Kernels:\n ",nn.conv_layer_2_kernels)
# nn.max_pooling(2)

# # max_pool_image = nn.conv_layer_max_pool_images
# # print(max_pool_image.shape)

# # print("\n")
# # print("Conv Layer 2 Kernels Shape: \n",nn.conv_layer_2_kernels.shape)
# # print("Conv Layer 2 Activation Shape: \n",nn.conv_layer_2_activation_images.shape)

# print("\n")
# # print("Conv Layer 2 Activation Image 1 Before: \n",nn.conv_layer_2_activation_images[0])
# nn.convulate_2(nn.conv_layer_max_pool_images, nn.conv_layer_2_kernels, 1)
# # print("Conv Layer 2 Activation Image 1 After: \n",nn.conv_layer_2_activation_images[0])

# print("Conv Layer 2 Activation Image 1  Shape: ",conv_layer_2_activation_images[3].shape)
# print("Conv Layer 2 Max Pooling Layers Shape: ",conv_layer_2_max_pooling_images[0].shape)

# nn.max_pooling(2)

# flattened_images = [image.flatten() for image in nn.conv_layer_2_max_pool_images]
# flattened_images = np.array(flattened_images)
# flattened_images = flattened_images.flatten()
# print("Flattened Images Shape: ",flattened_images.shape)











# def convulate_2(self, max_pool_images, kernel, stride):
    #     num_filters = kernel.shape[0]
    #     activation_image_row = self.conv_layer_2_activation_images.shape[0]
    #     activation_image_col = self.conv_layer_2_activation_images.shape[1]

    #     filters_across_images = max_pool_images.shape[2] # Number of filters altogether per images

    #     number_of_filters = int(kernel.shape[0] / filters_across_images)
    #     print("Number of Filters: ",number_of_filters)
    #     # Use 16 filters across each image 32 times
    #     kernel_index = 0
    #     for i in range(0, num_filters, filters_across_images):
    #         for j in range(activation_image_row):
    #             for k in range(activation_image_col):
    #                 # Calculate the starting and ending index of the patch for this filter
    #                 start_row = j * stride
    #                 end_row = start_row + kernel.shape[1]
    #                 start_col = k * stride
    #                 end_col = start_col + kernel.shape[2]

    #                 # Extract the current patch from the images
    #                 patches = []
    #                 for f in range(self.conv_layer_activation_images.shape[2]):
    #                     max_pool_image = max_pool_images[:, :, f]
    #                     patches.append(max_pool_image[start_row:end_row, start_col:end_col])
                    
    #                 filters = []
    #                 for l in range(filters_across_images):
    #                     filters.append(kernel[i + l][2])
                    
    #                 patches = np.array(patches)
    #                 filters = np.array(filters)
                    
    #                 conv_value = np.sum(patches * filters) + self.bias_conv_layer_2[0, kernel_index]

    #                 # Apply the Sigmoid activation function
    #                 relu_value = self.relu(conv_value)

    #                 self.conv_layer_2_activation_images[j, k, kernel_index] = relu_value

    #         kernel_index += 1
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # self.flattened_images = self.flattened_images.reshape(-1, 1)
        # print("Flattened Images : ",self.flattened_images)
        # print("Flattened Images Shape: ",self.flattened_images.shape)
        # output_error = self.cross_entropy(y, self.final_output)
        # output_derivative = self.sigmoid_derivative(self.final_output)
        # d_output_delta = self.calculate_delta(output_error, output_derivative)
        # print("Final Output: ",self.final_output)  
        # print("Target Output: ",y)   
        # print("Output Error: ",output_error)
        # print("Delta Output: ",d_output_delta)
        # d_output_change = np.dot(self.flattened_images, d_output_delta)  
        # delta_fully_connected_weights = np.dot(d_output_delta, self.flattened_images.T)
        # delta_bias_output = np.sum(d_output_delta)

        # # Update the weights and bias for the fully connected layer
        # d_flattened_image_delta = np.dot(d_output_delta, self.fully_connected_weights.T)
        # # print("Delta Flattened Image: ",d_flattened_image_delta)
        
        # weights_before = self.fully_connected_weights.copy()
        # print("Fully Connected Weights Before: ",weights_before)
        # self.fully_connected_weights -= learning_rate * d_output_change
        # self.bias_output -= learning_rate * np.sum(d_output_delta)
        # print("Fully Connected Weights After: ",self.fully_connected_weights)
        # print("Weights Change: ", np.sum(weights_before - self.fully_connected_weights))
        # 
        # 
        # 
        # 
        #  #     for i in range(0, num_samples, batch_size):
        #         batch_indices = indices[i:i+batch_size]
        #         X_batch = X[batch_indices]
        #         y_batch = y[batch_indices]
        #         current_batch_size = X_batch.shape[0]


        #         if current_batch_size < batch_size:
        #             self.conv_layer_activation_images = np.zeros((
        #                 current_batch_size, 
        #                 self.conv_layer_activation_images.shape[1], 
        #                 self.conv_layer_activation_images.shape[2], 
        #                 self.conv_layer_activation_images.shape[3]))
        #             self.conv_layer_max_pool_images = np.zeros((
        #                 current_batch_size, 
        #                 self.conv_layer_max_pool_images.shape[1], 
        #                 self.conv_layer_max_pool_images.shape[2], 
        #                 self.conv_layer_max_pool_images.shape[3]))
        #             self.conv_layer_2_activation_images = np.zeros((
        #                 current_batch_size, 
        #                 self.conv_layer_2_activation_images.shape[1], 
        #                 self.conv_layer_2_activation_images.shape[2], 
        #                 self.conv_layer_2_activation_images.shape[3]))
        #             self.conv_layer_2_max_pool_images = np.zeros((
        #                 current_batch_size, 
        #                 self.conv_layer_2_max_pool_images.shape[1], 
        #                 self.conv_layer_2_max_pool_images.shape[2], 
        #                 self.conv_layer_2_max_pool_images.shape[3]))
                
        #         start_time = time.time()
        #         output = self.forward_pass_batch(X_batch)
        #         loss = self.cross_entropy(y_batch, output)
        #         if loss < 0.01 or math.isnan(loss):
        #             print(f"Epoch {epoch}, Loss: {loss}")
        #             print("Training Completed")
        #             print("Exiting Training.....")
        #             return
                    

        #         if epoch % 1 == 0:
        #             print(f"Epoch {epoch}, Loss: {loss}")
        #             print()
        #         self.backpropagation_batch(X_batch, y_batch, learning_rate)


        #         print("Time Taken: ",time.time() - start_time)