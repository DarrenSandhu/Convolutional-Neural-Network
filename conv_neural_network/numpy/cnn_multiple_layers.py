import math
import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
import warnings
import time
from conv_neural_network.numpy.load_cnn_training_data_np import images, labels
from conv_neural_network.numpy.conv_layer_list_np import Convolutional_Layers

warnings.filterwarnings("ignore", message="Corrupt JPEG data")
BASE_DIR = os.path.dirname(os.path.abspath(__name__))



class Convulutional_Neural_Network():

    def __init__(self, input_nodes, output_nodes, conv_layers, fully_connected_weights=None, bias_output=None, bias_conv_layers=None):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        print("Input Nodes: ",self.input_nodes)
        print("Output Nodes: ",self.output_nodes)
        self.conv_layers = conv_layers
        self.last_conv_layer = conv_layers.getLastLayer()
        # Intialize weights and bias for the fully connected layer
        if fully_connected_weights is not None:
            self.fully_connected_weights = fully_connected_weights
        else:
            self.fully_connected_weights = np.random.randn(self.last_conv_layer.max_pool_images.shape[1] * self.last_conv_layer.max_pool_images.shape[2] * self.last_conv_layer.max_pool_images.shape[3], self.output_nodes) * np.sqrt(2 / (self.last_conv_layer.max_pool_images.shape[1] * self.last_conv_layer.max_pool_images.shape[2] * self.last_conv_layer.max_pool_images.shape[3] + self.output_nodes))
        print("Full Connected Weights Shape: ",self.fully_connected_weights.shape)
        if bias_output is not None:
            self.bias_output = bias_output
        else:
            self.bias_output = np.random.randn(1, self.output_nodes) * 0.01
        print("Output Bias Shape: ",self.bias_output.shape)

        # Intialize bias for all convolutional layers
        self.bias_conv_layers = []
        print("Conv Layers Size: ",self.conv_layers.size())
        for i in range(self.conv_layers.size()):
            self.bias_conv_layers.append(np.random.randn(1, self.conv_layers.getLayer(i).kernels.shape[0]) * 0.01)

        for i in range(len(self.bias_conv_layers)):
            print("Bias Conv Layer Shape: ",self.bias_conv_layers[i].shape) 

    
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
    
    def convolve_backprop_image_and_rotated_kernel(self, padded_activation_images, rotated_kernel, stride=1, filters_in_pooling_layer=None):
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
        backprop_pooled_images = np.zeros((batch_size, output_height, output_width, filters_in_pooling_layer))
        
        # Perform the convolution
        for f in range(batch_size):
            for k in range(num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract the patch from the padded activation images
                        patch = padded_activation_images[f, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width, :kernel_depth]
                        
                        # For each filter, convolve the patch with the rotated kernel
                        for k in range(filters_in_pooling_layer):  # Loop over filters in the kernel
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
        for i in range(self.conv_layers.size()):
            conv = self.conv_layers.getLayer(i)
            if i == 0:
                self.convulate_batch(X_batch, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            else:
                self.convulate_batch(self.conv_layers.getLayer(i-1).max_pool_images, conv.kernels, self.bias_conv_layers[i], conv.activation_images, conv.activation_func, 1)
            self.max_pooling_batch(2, conv.activation_images, conv.max_pool_images)

        batch_size = X_batch.shape[0]
        self.flattened_images = self.last_conv_layer.max_pool_images.reshape(batch_size, -1)
        print("Flattened Images Shape: ",self.flattened_images.shape)

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
        print("Output Derivative: ",output_derivative.shape)
        d_output_delta = output_error * output_derivative

        self.flattened_images = self.flattened_images.T
        
        # Update the weights using gradient descent (assuming no momentum, for simplicity)
        self.fully_connected_weights -= learning_rate * np.dot(self.flattened_images, d_output_delta)

        self.bias_output -= learning_rate * np.sum(d_output_delta)

        ############################################################################################################
        ############################################################################################################
        d_flattened_image_delta = np.dot(d_output_delta, self.fully_connected_weights.T)
        delta_loss_over_delta_max_pool_images = d_flattened_image_delta.reshape(self.last_conv_layer.max_pool_images.shape)

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the layers

        for i in range(self.conv_layers.size()-1, -1, -1):
            conv = self.conv_layers.getLayer(i)
            # Backpropagate through the max pooling layer
            backprop_derivative_images = self.backprop_max_pooling(delta_loss_over_delta_max_pool_images, conv.activation_images)
            # Backpropagate through the ReLU activation function to get the derivative
            conv_relu_derivative = self.backprop_calculate_relu_derivative(conv.activation_images)
            # Multiply the backpropagated derivative with the ReLU derivative to get the activation images of the previous layer
            backprop_activation_images = np.multiply(backprop_derivative_images, conv_relu_derivative)
            ############################################################################################################
            ############################################################################################################
            # Calculate the gradient of the kernels and biases
            d_kernel = np.zeros_like(conv.kernels)
            d_bias = np.zeros_like(self.bias_conv_layers[i])
            if i == 0:
                d_kernel, d_bias = self.backprop_convolution(d_kernel, d_bias, backprop_activation_images, X)
            else:
                d_kernel, d_bias = self.backprop_convolution(d_kernel, d_bias, backprop_activation_images, self.conv_layers.getLayer(i-1).max_pool_images)
            conv.kernels -= learning_rate * d_kernel
            self.bias_conv_layers[i] -= learning_rate * d_bias

            ############################################################################################################
            ############################################################################################################
            if i == 0:
                continue
            else:
                # Rotate the kernel by 180 degrees to convolve with the backpropagated activation images
                rotated_kernel = np.rot90(conv.kernels, 2)
                # print("Rotated Kernel: ",rotated_kernel)
                # Pad the backpropagated activation images to match the size of the max pool images of previous layer
                padded_backpropagated_activation_images = self.padding(backprop_activation_images, conv.kernels.shape[1] - 1)
                # print("Padded Backpropagated Activation Images: ",padded_backpropagated_activation_images.shape)
                # Convolve the padded backpropagated activation images with the rotated kernel
                start_time = time.time()
                backprop_pool_images = self.convolve_backprop_image_and_rotated_kernel(padded_backpropagated_activation_images, rotated_kernel, 1, self.conv_layers.getLayer(i-1).max_pool_images.shape[3])
                # print("Time Taken: ",time.time() - start_time)
                # print("Backprop Pool Images: ",backprop_pool_images.shape)
                delta_loss_over_delta_max_pool_images = backprop_pool_images


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
                y_batch = np.concatenate((y_cat[cat_batch_indices], y_dog[dog_batch_indices]), axis=0)
                batch_shuffle_indices = np.random.permutation(len(X_batch))

                X_batch = X_batch[batch_shuffle_indices]
                y_batch = y_batch[batch_shuffle_indices]

                current_batch_size = X_batch.shape[0]
                if current_batch_size < batch_size:
                    continue
                
                start_time = time.time()
                output = self.forward_pass_batch(X_batch)
                loss = self.cross_entropy(y_batch, output)

                if loss < 0.01 or math.isnan(loss):
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
                conv_layers=self.conv_layers,
                fully_connected_weights=self.fully_connected_weights,
                bias_output=self.bias_output,
                bias_conv_layer=self.bias_conv_layers)
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
print("Input Nodes: ",input_nodes)
output_nodes = 2
target = labels
batch_size = 2
conv_layers = Convolutional_Layers(2, [[8,5], [16,3]], ['relu', 'relu'], images.shape, batch_size)
for i in range(len(conv_layers.conv_layers)):
    print("Layer: ",i)
    print("Kernels: ",conv_layers.conv_layers[i].kernels.shape)
    print("Activation Images: ",conv_layers.conv_layers[i].activation_images.shape)
    print("Max Pool Images: ",conv_layers.conv_layers[i].max_pool_images.shape)
    print("Activation Function: ",conv_layers.conv_layers[i].activation_func)
    print("\n")

print("\n")
nn = Convulutional_Neural_Network(
    input_nodes, 
    output_nodes,
    conv_layers,
    )
# start_time = time.time()
# nn.forward_pass_batch(images[:batch_size])
# print("Time Taken: ",time.time() - start_time)
# # print("Output Shape: ",output.shape)
# # print("Output: ",output)
# start_time = time.time()
# nn.backpropagation_batch(images[:batch_size], target[:batch_size], 0.001)
# print("Time Taken: ",time.time() - start_time)
images = np.concatenate((images[:6], images[1000:1006]), axis=0)
target = np.concatenate((target[:6], target[1000:1006]), axis=0)
nn.train(images, target, 100, 0.01, batch_size)


# nn.save('trained_cnn_model.npz')
# prediction_image = cv2.imread(os.path.join(BASE_DIR, 'training_data', 'cat', '10580.jpg'))
# prediction_image = cv2.resize(prediction_image, (64, 64))
# prediction_image = np.array([prediction_image]) / 255.0
# prediction = nn.predict(prediction_image)
# print("Prediction: ",prediction)
