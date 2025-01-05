import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
from PIL import UnidentifiedImageError, Image
import warnings
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
print(BASE_DIR)

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, hidden_nodes, weights_input_to_hidden=None, weights_hidden_to_output=None):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes


        if weights_input_to_hidden is not None:
            self.weights_input_to_hidden = weights_input_to_hidden
        else:
            self.weights_input_to_hidden = np.random.randn(self.input_nodes, self.hidden_nodes)
        print("Hidden Weights Shapse:",self.weights_input_to_hidden.shape)
        if weights_hidden_to_output is not None:
            self.weights_hidden_to_output = weights_hidden_to_output
        else:
            self.weights_hidden_to_output = np.random.randn(self.hidden_nodes, self.output_nodes)
        print("Output Weights Shape:",self.weights_hidden_to_output.shape)
        print("\n")
        # print("Weights For Hidden: \n",self.weights_input_to_hidden)
        # print("Weights For Output: \n", self.weights_hidden_to_output)

        # self.bias_hidden = np.full((1, self.hidden_nodes), 0.35)
        self.bias_hidden = np.random.randn(1, self.hidden_nodes)
        print("Hidden Bias Shape: ",self.bias_hidden.shape) 
        print("Hidden Bias: ",self.bias_hidden)
        print("\n")
        # self.bias_output = np.full((1, self.output_nodes), 0.60)
        self.bias_output = np.random.randn(1, self.output_nodes)
        print("Output Bias Shape: ",self.bias_output.shape)
        print("Output Bias: ",self.bias_output)
        print("\n")
        # print("\nBias For Hidden: ",self.bias_hidden)
        # print("Bias For Output: ",self.bias_output)
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
    
    def derivative(self, output):
        return output * (1 - output)
    
    def update_weight(self, weight, learning_rate, delta):
        weight = weight - (learning_rate * delta)
        return weight
    
    def weighted_sum(self, weight, value):
        return np.dot(weight, value)
    
    def target_output_diff(self, target_output, output):
        return target_output - output
    
    def squared_error(self, target_output, output):
        return np.power((target_output - output), 2)
    
    def calculate_error(self, target_output, final_output):
        return final_output - target_output
    
    def calculate_delta(self, error, derivative):
        return error * derivative
    
    def mean_squared_error(self, target_outputs, outputs):
        squared_errors = self.squared_error(target_outputs, outputs)
        self.mse = np.mean(squared_errors)
        return self.mse 
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        # print("Hidden Input Shape: ", self.hidden_input.shape)
        # self.hidden_output = # Replace sigmoid with ReLU in your forward pass
        self.hidden_output = self.sigmoid(self.hidden_input)
        # print("Hidden Output Shape: ", self.hidden_output.shape)

        self.final_input = self.weighted_sum(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        # print("Final Input Shape: ", self.final_input.shape)
        self.final_output = self.sigmoid(self.final_input)
        # print("Final Output Shape: ", self.final_output.shape)

        return self.final_output

        

    
    def backpropagation(self, X, y, learning_rate):
        # y = y.reshape(-1, 1)
        self.output_error = self.calculate_error(y, self.final_output)
        self.output_derivative = self.derivative(self.final_output)
        self.output_delta = self.calculate_delta(self.output_error, self.output_derivative)

        self.output_change = np.dot(self.hidden_output.T, self.output_delta)
        # print("Output Change Shape: ", self.output_change.shape)
        
        
        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_to_output.T)
        self.hidden_delta = self.calculate_delta(self.hidden_error, self.derivative(self.hidden_output))
        

        self.weights_hidden_to_output -= (self.output_change * learning_rate)
        self.weights_input_to_hidden -= (np.dot(X.T, self.hidden_delta) * learning_rate)

        self.bias_output -= learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.bias_hidden -= learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)


    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")

# Load the training data for cats
# cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
# cat_directory_files = os.listdir(cat_directory)

# images = []
# labels = []
# image_size = (64, 64)
# for filename in cat_directory_files:
#     img_path = os.path.join(cat_directory, filename)
#     assert os.path.exists(img_path), f"File not found: {img_path}"
#     try:
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, image_size)
#         images.append(img)
#         labels.append(1)
#     except Exception as e:
#         # print(f"Error processing image {img_path}: {e}")
#         continue  # Skip corrupt images

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

# print("Images Size: ", len(images)) 
# print("Labels Size: ", len(labels))
# images = np.array(images)
# labels = np.array(labels)
# # print("Images: ", images)

# # Normalize the image data
# images = images / 255.0
# print("Images: ", images)
# print("Labels: ", labels)

# print("Images Shape: ", images.shape)
# print("Labels Shape: ", labels.shape)



# example_matrix = np.random.rand(2, 4, 5, 3)

# print("Example matrix shape:", example_matrix.shape)
# print("Example matrix:\n", example_matrix)

# Test cat image
cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
image_size = (64, 64)

cat_image = cv2.imread(os.path.join(cat_directory, '1.jpg'))
cat_image = cv2.resize(cat_image, image_size)

cat_image_2 = cv2.imread(os.path.join(cat_directory, '2.jpg'))
cat_image_2 = cv2.resize(cat_image_2, image_size)

# Normalize the image data
images = []
images.append(cat_image)
images.append(cat_image_2)
images = np.array(images) / 255.0
print("Images: ", images)

# Create labels for the images
labels = np.array([1, 1])
print("Labels: ", labels)
probabilities = np.array([0, 1],[0, 1])

print("Images Shape: ", images.shape)


# Create a neural network
input_nodes = 64 * 64 * 3
output_nodes = 2
hidden_nodes = 10
nn = Neural_Network(input_nodes, output_nodes, hidden_nodes)

# Train the neural network
nn.train(images, probabilities, epochs=1000, learning_rate=0.1)

