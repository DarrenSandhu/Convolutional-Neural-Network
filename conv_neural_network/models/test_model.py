from keras._tf_keras.keras.models import load_model
from get_validation_data_torch import cat_images, cat_labels, dog_images, dog_labels

# Load the model
model = load_model("cnn_model_2.h5")

# Evaluate the model
cat_loss, cat_accuracy = model.evaluate(cat_images, cat_labels)
dog_loss, dog_accuracy = model.evaluate(dog_images, dog_labels)

print(f"Cat loss: {cat_loss}, Cat accuracy: {cat_accuracy}")
print(f"Dog loss: {dog_loss}, Dog accuracy: {dog_accuracy}")