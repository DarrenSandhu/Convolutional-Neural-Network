import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
from PIL import UnidentifiedImageError, Image
import warnings
import time
from conv_neural_network.numpy.conv_layer_list_np import Convolutional_Layers
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

class Convulutional_Neural_Network():

    def __init__(self, input_nodes, output_nodes, conv_layer_kernels, conv_layer_activation_images, conv_layer_max_pool_images, conv_layer_2_activation_images=None, conv_layer_2_kernels=None, conv_layer_2_max_pool_images=None):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.conv_layer_kernels = conv_layer_kernels
        self.conv_layer_activation_images = conv_layer_activation_images
        self.conv_layer_max_pool_images = conv_layer_max_pool_images
        self.conv_layer_2_activation_images = conv_layer_2_activation_images
        self.conv_layer_2_kernels = conv_layer_2_kernels
        self.conv_layer_2_max_pool_images = conv_layer_2_max_pool_images

        # Intialize weights and bias for the fully connected layer
        self.fully_connected_weights = np.random.randn(self.conv_layer_2_max_pool_images.shape[0] * self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2], self.output_nodes) * np.sqrt(2 / (self.conv_layer_2_max_pool_images.shape[0] * self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2] + self.output_nodes))
        print("Full Connected Weights Shape: ",self.fully_connected_weights.shape)
        self.bias_output = np.random.randn(1, self.output_nodes) * np.sqrt(2 / (self.conv_layer_2_max_pool_images.shape[0] * self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2] + self.output_nodes))
        print("Output Bias Shape: ",self.bias_output.shape)


        # Intialize bias for the convolutional layer
        self.bias_conv_layer = np.random.randn(1, self.conv_layer_kernels.shape[0]) * 0.01
        
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
    
    def convulate(self, image, kernel, bias, activation_image, activation_function, stride):
        num_filters = kernel.shape[0]
        activation_image_row = activation_image.shape[0]
        activation_image_col = activation_image.shape[1]
        # print("Activation Image Shape: ",activation_image.shape)
        # print("Image Shape: ",image.shape)
        # print("Kernel Shape: ",kernel.shape)
        # print("Bias Shape: ",bias.shape)
        # print("Num Filters: ",num_filters)
        # print()

        for i in range(num_filters):
            for j in range(activation_image_row):
                for k in range(activation_image_col):
                    # Calculate the starting and ending index of the patch for this filter
                    start_row = j * stride
                    end_row = start_row + kernel.shape[1]
                    start_col = k * stride
                    end_col = start_col + kernel.shape[2]

                    patch = image[start_row:end_row, start_col:end_col, :kernel.shape[3]]
            
                    conv_value = np.sum(patch * kernel[i]) + bias[0, i]

                    # Apply the ReLU activation function
                    if activation_function == 'relu':
                        relu_value = self.relu(conv_value)
                    else:
                        relu_value = self.sigmoid(conv_value)
    
                    activation_image[j, k, i] = relu_value
 
    def max_pooling(self, stride, activation_images, max_pool_images):
        pool_size = 2  # Typically, max pooling uses a 2x2 pool size
        # Determine the shape of the output max-pooling images
        num_filters = activation_images.shape[2]

        for i in range(num_filters):
            for j in range(0, activation_images.shape[0] - pool_size + 1, stride):
                for k in range(0, activation_images.shape[1] - pool_size + 1, stride):
                    # Extract the patch of size (2, 2, depth) for pooling
                    patch = activation_images[j:j+pool_size, k:k+pool_size, i]

                    # Calculate the max of the patch
                    max_value = np.average(patch)

                    # Store the result in the max pooling layer
                    max_pool_images[j // stride, k // stride, i] = max_value
    
    
        


    def flatten(self, images):
        flattened_images = [image.flatten() for image in images]
        flattened_images = np.array(flattened_images)
        flattened_images = flattened_images.flatten()
        return flattened_images
    
    def padding(self, image, padding_size):
        padded_image = np.zeros((image.shape[0] + 2 * padding_size, image.shape[1] + 2 * padding_size, image.shape[2]))
        padded_image[padding_size:-padding_size, padding_size:-padding_size] = image
        return padded_image
    
    def rotate_np_image(self, image, angle):
        return np.rot90(image, 2)

    def backprop_convolution(self, d_kernel, d_bias, backprop_activation_images, patch_extraction_images):

        # Iterate over the number of filters
        # print()
        # print("D Kernel Shape: ",d_kernel.shape)
        # print("Backprop Activation Images Shape: ",backprop_activation_images.shape)
        # print("Patch Extraction Images Shape: ",patch_extraction_images.shape)
        for filter_index in range(d_kernel.shape[0]):
            # Slide over the spatial dimensions of backprop_activation_images
            for i in range(backprop_activation_images.shape[0]):
                for j in range(backprop_activation_images.shape[1]):
                    # print("J: ",j)
                    # Define the patch of the max pool images corresponding to this position
                    start_i = i  # Adjust for stride if necessary
                    end_i = start_i + d_kernel.shape[1]
                    start_j = j  # Adjust for stride if necessary
                    end_j = start_j + d_kernel.shape[2]

                    # Extract the patch
                    patch = patch_extraction_images[start_i:end_i, start_j:end_j, :d_kernel.shape[3]]

                    # Accumulate gradient for the kernel
                    # print("D Kernel: ", d_kernel.shape)
                    # print("Backprop Activation Images: ",backprop_activation_images.shape)
                    d_kernel[filter_index] += backprop_activation_images[i, j, filter_index] * patch

            # Accumulate gradient for the bias
            d_bias[0, filter_index] += np.sum(backprop_activation_images[:, :, filter_index])

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
        height, width, depth = padded_activation_images.shape
        
        # Calculate the output dimensions
        output_height = (height - kernel_height) // stride + 1
        output_width = (width - kernel_width) // stride + 1
        
        # Initialize the gradient (this will be the result of the convolution)
        backprop_pooled_images = np.zeros((output_height, output_width, self.conv_layer_max_pool_images.shape[2]))
        
        # Perform the convolution
        
        for i in range(output_height):
            for j in range(output_width):
                # Extract the patch from the padded activation images
                patch = padded_activation_images[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width, :kernel_depth]
                
                # For each filter, convolve the patch with the rotated kernel
                for k in range(self.conv_layer_max_pool_images.shape[2]):
                    backprop_pooled_images[i, j, k] = np.sum(patch * rotated_kernel[k])  # Sum over depth and kernel

        # print("Backprop Pooled Images Shape: ",backprop_pooled_images.shape)
        return backprop_pooled_images
    
    def backprop_relu(self, d_relu, activation_images):
        """
        Backpropagate through the ReLU activation function.
        :param d_relu: Gradient from the upper layer.
        :param activation_images: Activations before the ReLU function.
        """
        d_activation = np.where(activation_images > 0, d_relu, 0)
        return d_activation

    def backprop_max_pooling(self, d_pooled, activation_images, pool_size=2):
        """
        Backpropagate through the pooling layer.
        :param d_pooled: Gradient from the upper layer.
        :param activation_images: Activations before pooling.
        """
        d_activation_layer = np.zeros_like(activation_images)

        print("Activation Layer Shape: ",d_activation_layer.shape)
        print("Pooled Layer Shape: ",d_pooled.shape)
        for i in range(d_pooled.shape[2]):
            for j in range(d_pooled.shape[0]):
                for k in range(d_pooled.shape[1]):
                    # Get the pooling region in the activation image
                    h_start, h_end = j * pool_size, (j + 1) * pool_size
                    w_start, w_end = k * pool_size, (k + 1) * pool_size
                    
                    # Get pooling region where the max value occurred
                    pooling_region = activation_images[h_start:h_end, w_start:w_end, i]

                    # Get the max value location
                    max_val = np.max(pooling_region)
                    mask = pooling_region == max_val            

                    # Update the new activation layer
                    d_activation_layer[h_start:h_end, w_start:w_end, i] = d_pooled[j, k, i] * mask

        return d_activation_layer

    def backprop_calculate_relu_derivative(self, activation_images):
        """
        Calculate the derivative of the ReLU activation function.
        :param activation_images: Activations before the ReLU function.
        """
        return np.where(activation_images > 0, 1, 0)
    
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
        
        # Initialize an empty array for the activation images (output of convolution)
        activation_images = np.zeros((batch_size, height - kernel_height + 1, width - kernel_width + 1, num_kernels))
        
        # Apply convolution to each image in the batch
        for i in range(batch_size):
            img = X_batch[i]
            for k in range(num_kernels):
                kernel = kernels[k]
                bias = biases[k]
                
                # Apply the convolution for this image and kernel
                for h in range(0, height - kernel_height + 1, stride):
                    for w in range(0, width - kernel_width + 1, stride):
                        # Extract the current patch of the image
                        patch = img[h:h+kernel_height, w:w+kernel_width, :]
                        
                        # Convolve (element-wise multiplication and sum)
                        conv_result = np.sum(patch * kernel) + bias
                        
                        # Store the result in the activation map
                        activation_images[i, h // stride, w // stride, k] = conv_result
        
        # Apply the activation function (ReLU, Sigmoid, etc.)
        if activation_func == 'relu':
            activation_images = np.maximum(0, activation_images)
        elif activation_func == 'sigmoid':
            activation_images = 1 / (1 + np.exp(-activation_images))
        
        return activation_images

    def forward_pass(self, X):
        self.convulate(X, self.conv_layer_kernels, self.bias_conv_layer, self.conv_layer_activation_images, 'relu', 1)
        self.max_pooling(2, self.conv_layer_activation_images, self.conv_layer_max_pool_images)
        self.convulate(self.conv_layer_max_pool_images, self.conv_layer_2_kernels, self.bias_conv_layer_2, self.conv_layer_2_activation_images, 'relu', 1)
        self.max_pooling(2, self.conv_layer_2_activation_images, self.conv_layer_2_max_pool_images)
        # print("Conv Layer 2 Max Pooling Images Shape: ",self.conv_layer_2_max_pool_images.shape)
        self.flattened_images = self.conv_layer_2_max_pool_images.flatten().reshape(-1)
        # print("Flattened Images Shape: ",self.flattened_images.shape)
        # Fully connected layer
        self.final_input = np.dot(self.flattened_images, self.fully_connected_weights) + self.bias_output
        # print("Final Input: ",self.final_input)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backpropagation(self, X, y, learning_rate):
        ############################################################################################################
        ############################################################################################################
        # Forward pass
        print("Output: ",self.final_output)
        print("Target: ",y)
        
        
        # Calculate the output error
        output_error = self.final_output - y  # For binary classification, this is the error (prediction - true)
        
        # Calculate the gradient of the output layer
        output_derivative = self.sigmoid_derivative(self.final_output)
        d_output_delta = output_error * output_derivative
        # print("D Output Delta : ",d_output_delta)

        # Ensure self.flattened_images.T has the shape (3136, 1)
        self.flattened_images = self.flattened_images.reshape(-1, 1)
        
        # Update the weights using gradient descent (assuming no momentum, for simplicity)
        weights_before = self.fully_connected_weights.copy()
        # print("Fully Connected Weights Before: ",weights_before)
        self.fully_connected_weights -= learning_rate * np.dot(self.flattened_images, d_output_delta)
        # print("Fully Connected Weights After: ",self.fully_connected_weights)
        print("Fully Connected Weights Change: ", np.sum(weights_before - self.fully_connected_weights))
        self.bias_output -= learning_rate * np.sum(d_output_delta)

        # ############################################################################################################
        # ############################################################################################################
        d_flattened_image_delta = np.dot(d_output_delta, self.fully_connected_weights.T)
        # print("D Flattened Image Delta Shape: ",d_flattened_image_delta.shape)
        delta_loss_over_delta_max_pool_2_images = d_flattened_image_delta.reshape(self.conv_layer_2_max_pool_images.shape)
        # print("Delta Loss Over Delta Max Pool 2 Images Shape: ",delta_loss_over_delta_max_pool_2_images.shape)
        

        ############################################################################################################
        ############################################################################################################
        # # Backpropagate through the second pooling layer

        # # Calculate the gradient of the loss with respect to the relu activation
        backprop_derivative_images_layer_2 = self.backprop_max_pooling(delta_loss_over_delta_max_pool_2_images, self.conv_layer_2_activation_images)
        # print("Backpropagated Derivative Images Layer 2 Shape: ",backprop_derivative_images_layer_2.shape)

        # Calculate the gradient of the loss with respect to the convultional layer 2 images
        conv_layer_2_relu_derivative = self.backprop_calculate_relu_derivative(self.conv_layer_2_activation_images)
        # print("Conv Layer 2 ReLU Derivative: ",conv_layer_2_relu_derivative)
        backprop_activation_images_layer_2 = np.multiply(backprop_derivative_images_layer_2, conv_layer_2_relu_derivative)
        # print("Backpropagated Activation Images Shape: ",backprop_activation_images_layer_2.shape)

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the second convolutional layer

        # Calculate the gradient of the loss with respect to the kernel and bias
        d_kernel_2 = np.zeros_like(self.conv_layer_2_kernels)
        d_bias_2 = np.zeros_like(self.bias_conv_layer_2)
        d_kernel_2, d_bias_2 = self.backprop_convolution(d_kernel_2, d_bias_2, backprop_activation_images_layer_2, self.conv_layer_max_pool_images)

        # Update the kernel and bias for the second convolutional layer
        kernel_before = self.conv_layer_2_kernels.copy()
        self.conv_layer_2_kernels -= learning_rate * d_kernel_2
        print("Kernel Layer 2 Weights Change: ",np.sum(kernel_before - self.conv_layer_2_kernels))
        self.bias_conv_layer_2 -= learning_rate * d_bias_2

        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the first pooling layer

        # Rotate the kernel by 180 degrees
        rotated_kernel = np.rot90(self.conv_layer_2_kernels, 2)

        # Add padding to the backpropagated activation images
        padding_size = self.conv_layer_2_kernels.shape[1] - 1
        # print("Padding Size: ",padding_size)
        padded_backpropagated_activation_images = self.padding(backprop_activation_images_layer_2, padding_size)
        # print("Padded Backpropagated Activation Images Shape: ",padded_backpropagated_activation_images.shape)

        # Convolve the padded activation images with the rotated kernel
        backprop_pool_2_images = self.convolve_backprop_image_and_rotated_kernel(padded_backpropagated_activation_images, rotated_kernel, 1)
        # print("Backprop Pool 2 Images Shape: ",backprop_pool_2_images.shape)


        ############################################################################################################
        ############################################################################################################
        # Backpropagate through the first convolutional layer

        # Calculate backpropagation through the ReLU activation function
        # Change backprop_pool_2_images to backprop_activation_images_layer_1
        # backprop_derivative_activation_images_layer_1 = self.backprop_max_pooling(backprop_pool_2_images, self.conv_layer_activation_images)
        # print("Backprop ReLU Images Shape: ", backprop_derivative_activation_images_layer_1.shape)


        # # Calculate the gradient of the loss with respect to the relu activation
        # conv_layer_relu_derivative = self.backprop_calculate_relu_derivative(self.conv_layer_activation_images)
        # print("Conv Layer 1 ReLU Derivative Shape: ",conv_layer_relu_derivative.shape)

        # # Calculate the gradient of the loss with respect to the activation images
        # backprop_activation_images_layer_1 = np.multiply(backprop_derivative_activation_images_layer_1, conv_layer_relu_derivative)
        # print("Backprop Activation Images Layer 1 Shape: ",backprop_activation_images_layer_1.shape)


        # ############################################################################################################
        # ############################################################################################################
        # # Get the kernel and bias for the first convolutional layer
        # d_kernel_1 = np.zeros_like(self.conv_layer_kernels)
        # d_bias_1 = np.zeros_like(self.bias_conv_layer)
        # d_kernel_1, d_bias_1 = self.backprop_convolution(d_kernel_1, d_bias_1, backprop_activation_images_layer_1, X)


        # # Update the kernel and bias for the first convolutional layer
        # kernel_layer_1_before = self.conv_layer_kernels.copy()
        # self.conv_layer_kernels -= learning_rate * d_kernel_1
        # print("Kernel Layer 1 Weights change: ",np.sum(kernel_layer_1_before - self.conv_layer_kernels))
        # self.bias_conv_layer -= learning_rate * d_bias_1
        # print()
        # print()

    def train(self, X, y, epochs, learning_rate, batch_size=32):
        """
        Train the neural network using the specified training data.
        :param X: The images data.
        :param y: The labels for the images.
        :param epochs: The number of epochs to train the network.
        :param learning_rate: The learning rate for the network.
        :param batch_size: The number of images to use in each batch.
        """

        for epoch in range(epochs):
            output = self.forward_pass(X)
            loss = self.cross_entropy(y, output)
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            self.backpropagation(X, y, learning_rate)
                    
        
    
            
            
# Load the training data for cats

# try:
#     images = np.load('cat_images.npy')
#     labels = np.load('cat_labels.npy')
#     print("Loaded images and labels from file")
# except Exception as e:
#     cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
#     cat_directory_files = os.listdir(cat_directory)

#     images = []
#     labels = []
#     image_size = (64, 64)
#     for filename in cat_directory_files:
#         img_path = os.path.join(cat_directory, filename)
#         assert os.path.exists(img_path), f"File not found: {img_path}"
#         try:
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, image_size)
#             images.append(img)
#             labels.append([1, 0])
#         except Exception as e:
#             # print(f"Error processing image {img_path}: {e}")
#             continue  # Skip corrupt images
#     images = np.array(images) / 255.0
#     labels = np.array(labels)
#     np.save('cat_images.npy', images)
#     np.save('cat_labels.npy', labels)

# print("Images Shape: ",images.shape)
# print("Labels Shape: ",labels.shape)


# Test cat image
cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
image_size = (64, 64)

cat_image = cv2.imread(os.path.join(cat_directory, '10.jpg'))
cat_image = cv2.resize(cat_image, image_size)

# cat_image_2 = cv2.imread(os.path.join(cat_directory, '2.jpg'))
# cat_image_2 = cv2.resize(cat_image_2, image_size)

# Normalize the image data and labels data
images = []
labels = []

images.append(cat_image)
# images.append(cat_image_2)

labels.append([1, 0])
# labels.append([1, 0])

images = np.array(cat_image) / 255.0
print("Images: ", images.shape)

# Create labels for the images
labels = np.array(labels)
# print("Labels: ", labels)

# # print("Images Shape: ", images.shape)



###########################################
# Convolutional Neural Network Parameters #
###########################################
input_nodes = images.shape[0] * images.shape[1] * images.shape[2] # Number of input nodes equals the number of pixels in an image
output_nodes = 2
target = [1, 0]

print("Input Nodes: ", input_nodes)


############################################
# Convolutional Layer 1 Parameters #
############################################
conv_layer_kernels = np.random.randn(3, 5, 5, images.shape[2]) * np.sqrt(2 / (5 * 5 * images.shape[2]))
conv_layer_activation_images = (
np.zeros((
    images.shape[0] - conv_layer_kernels.shape[1] + 1,
    images.shape[1] - conv_layer_kernels.shape[2] + 1,
    conv_layer_kernels.shape[0]
)))   

conv_layer_max_pooling_images = (
np.zeros((
    conv_layer_activation_images.shape[0] // 2,
    conv_layer_activation_images.shape[1] // 2,
    conv_layer_activation_images.shape[2],
)))


############################################
# Convolutional Layer 2 Parameters #
############################################
conv_layer_2_kernels = np.random.randn(6, 3, 3, conv_layer_max_pooling_images.shape[2]) * np.sqrt(2 / (3 * 3 * conv_layer_max_pooling_images.shape[2]))
conv_layer_2_activation_images = (
np.zeros((
    conv_layer_max_pooling_images.shape[0] - conv_layer_2_kernels.shape[1] + 1,
    conv_layer_max_pooling_images.shape[1] - conv_layer_2_kernels.shape[2] + 1,
    conv_layer_2_kernels.shape[0]
)))

conv_layer_2_max_pooling_images = (
np.zeros((
    conv_layer_2_activation_images.shape[0] // 2,
    conv_layer_2_activation_images.shape[1] // 2,
    conv_layer_2_activation_images.shape[2]
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

# # start_time = time.time()
# final_output = nn.forward_pass(images)
# backprop = nn.backpropagation(images, target, 0.01)
# print("Time Taken For One Epoch: ",time.time() - start_time)
nn.train(images, target, 10000, 0.5)


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