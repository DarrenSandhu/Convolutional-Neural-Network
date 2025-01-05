import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_methods.load_trained_cnn_torch import open_trained_torch_cnn, open_trained_torch_multiple_fc_cnn
from data_methods.get_validation_data_torch import cat_images, cat_labels, dog_images, dog_labels
from data_methods.get_cnn_training_data_torch import images, labels
import torch

# Load the trained model
filename = "trained_cnn_model.pth"
cnn = open_trained_torch_multiple_fc_cnn(filename)
batch_size = cnn.conv_layers[0].batch_size

images = torch.cat((images[:10], images[6000:6500]), dim=0)
labels = torch.cat((labels[:10], labels[6000:6500]), dim=0)
# cnn.train(images, labels, 1000, 0.001, batch_size)
# batch_size = cnn.conv_layers[0].batch_size
# Test the model
print("Cat Images Shape: ",cat_images.shape)
predictions = cnn.predict(cat_images, batch_size)
# print(f"Predictions: {predictions}")
# Compare the predictions with the actual labels
correct = 0
print("Predictions Shape: ",len(predictions))
for i in range(len(predictions)):
    if i < cat_images.shape[0]:
        prediction = (predictions[i] > 0.5).float()
        # print(f"Prediction: {prediction}")
        if prediction == 1:
            correct += 1

accuracy = correct / len(predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

predictions2 = cnn.predict(dog_images, batch_size)
# print(f"Predictions: {predictions}")
# Compare the predictions with the actual labels
print("Predictions Shape: ",len(predictions2))
correct = 0
for i in range(len(predictions2)):
    if i < dog_images.shape[0]:
        prediction = (predictions2[i] > 0.5).float()
        # print(f"Prediction: {prediction}")
        if prediction == 0:
            correct += 1
        

accuracy = correct / len(predictions2)
print(f"Accuracy: {accuracy*100:.2f}%")
# print(f"Predictions: {predictions2}")

# # Test the model with training data
# # Shorten the training data for testing
# images = torch.cat((images[0:16], images[1000:1016]))
# labels = torch.cat((labels[0:16], labels[1000:1016]))
# predictions = cnn.predict(images, 8)
# print(f"Predictions: {predictions}")
# # Compare the predictions with the actual labels
# correct = 0
# for i in range(len(predictions)):
#     if torch.argmax(predictions[i]) == torch.argmax(labels[i]):
#         correct += 1

# accuracy = correct / len(predictions)
# print(f"Accuracy: {accuracy*100:.2f}%")

