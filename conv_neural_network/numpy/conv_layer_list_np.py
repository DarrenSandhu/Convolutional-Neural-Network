from conv_neural_network.numpy.conv_layer_np import Convolutional_Layer

class Convolutional_Layers():
    def __init__(self, layers, kernels, activation_func, image_shape, batch_size):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.layers = layers
        if len(kernels) != layers:
            raise ValueError("The number of kernels must match the number of layers")
        if len(activation_func) != layers:
            raise ValueError("The number of activation functions must match the number of layers")
        
        self.conv_layers = []
        prev_layer = None
        for i in range(layers):
            if i == 0:
                conv_layer = Convolutional_Layer(kernels[i], activation_func[i], image_shape, batch_size)
                prev_layer = conv_layer
                self.conv_layers.append(conv_layer)
            else:
                conv_layer = Convolutional_Layer(kernels[i], activation_func[i], prev_layer.max_pool_images.shape, batch_size)
                prev_layer = conv_layer
                self.conv_layers.append(conv_layer)
    
    def getLayer(self, index):
        return self.conv_layers[index]

    def getLastLayer(self):
        return self.conv_layers[-1]
    
    def size(self):
        return len(self.conv_layers)
    
