import numpy as np

class Convolutional_Layer():
    def __init__(self, kernels, activation_func, image_shape, batch_size):
        self.kernels = np.random.randn(kernels[0], kernels[1], kernels[1], image_shape[3]) * np.sqrt(2 / (5 * 5 * image_shape[3]))
        self.activation_func = activation_func
        self.activation_images = np.zeros((
            batch_size,
            image_shape[1] - self.kernels.shape[1] + 1,
            image_shape[2] - self.kernels.shape[2] + 1,
            self.kernels.shape[0]
        ))
        self.max_pool_images = np.zeros((
            batch_size,
            self.activation_images.shape[1] // 2,
            self.activation_images.shape[2] // 2,
            self.activation_images.shape[3]
        ))