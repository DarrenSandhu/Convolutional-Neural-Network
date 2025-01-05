from architecture.conv_layer_torch import Convolution_Layer_Torch

class Convolutional_Layers_Torch():
    def __init__(self, layers, kernels, activation_func, image_shape, batch_size, strides, weight_init):
        if len(kernels) != layers:
            raise ValueError("The number of kernels must match the number of layers")
        if len(activation_func) != layers:
            raise ValueError("The number of activation functions must match the number of layers")
        if len(strides) != layers:
            raise ValueError("The number of strides must match the number of layers")
        
        self.conv_layers = []
        prev_layer = None
        for i in range(layers):
            if i == 0:
                conv_layer = Convolution_Layer_Torch(kernels[i], activation_func[i], image_shape, batch_size, strides[i], weight_init)
                prev_layer = conv_layer
                self.conv_layers.append(conv_layer)
            else:
                conv_layer = Convolution_Layer_Torch(kernels[i], activation_func[i], prev_layer.max_pool_images.shape, batch_size, strides[i], weight_init)
                prev_layer = conv_layer
                self.conv_layers.append(conv_layer)

    def __iter__(self):
        return iter(self.conv_layers)
    
    def __getitem__(self, index):
        return self.conv_layers[index]
    
    def getLayer(self, index):
        return self.conv_layers[index]

    def getLastLayer(self):
        return self.conv_layers[-1]
    
    def size(self):
        return len(self.conv_layers)