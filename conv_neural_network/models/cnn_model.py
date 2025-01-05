import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from  get_cnn_training_data_torch import images, labels
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an image data generator with augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Convert the images and labels to a numpy array
labels = labels.cpu().numpy()
images = images.cpu().numpy()

# Create the model
model = models.Sequential()

# Add the first convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], images.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))

# Add the second convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# # Add the third convolutional layer
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

# # Add the fourth convolutional layer
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add a dense layer
model.add(layers.Flatten())

# Num of features in the output of the last conv layer
# Compute the number of features dynamically
# Build the model with a dummy input to get the shape
model.build((None, 128, 128, 3))  # Input shape
dummy_input = tf.random.normal([1, 128, 128, 3])
flattened_output = model(dummy_input)
num_features = flattened_output.shape[1]  # Number of features after flattening

print("Num Features: ", num_features)

# Add a fully connected layer
model.add(layers.Dense(512, activation='relu'))

# Add another fully connected layer
# model.add(layers.Dense(num_features, activation='relu'))

# Add dropout to the fully connected layer
model.add(layers.Dropout(0.5))

# Add the output layer
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Augment the training data
datagen.fit(X_train)

# Loss convergence
early_stopping = EarlyStopping(monitor='loss', patience=20)

# Train the model
model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save("cnn_model_2.h5")