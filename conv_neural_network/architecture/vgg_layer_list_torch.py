from architecture.vgg_layer_torch import VGG_Layer_Torch

class VGG_Layers_Torch():
    def __init__(self, layers, kernels, activation_func, image_shape, batch_size, conv_block, strides, weight_init=None):
        if len(kernels) != layers:
            raise ValueError("The number of kernels must match the number of layers")
        if len(activation_func) != layers:
            raise ValueError("The number of activation functions must match the number of layers")
        if len(strides) != layers:
            raise ValueError("The number of strides must match the number of layers")
        if len(conv_block) != layers:
            raise ValueError("The number of convolutional blocks must match the number of layers")
        self.conv_blocks = []
        self.conv_layers = []
        prev_layer = None
        for i in range(layers):
            temp = []
            if i == 0:
                for j in range(conv_block[i]):

                    if j == 0:
                        if j == conv_block[i] - 1:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], image_shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=True, weight_init=weight_init)
                        else:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], image_shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=False)
                            prev_conv_layer = conv_layer
                    else:
                        if j == conv_block[i] - 1:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_conv_layer.activation_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=True, weight_init=weight_init)
                        else:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_conv_layer.activation_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=False, weight_init=weight_init)
                            prev_conv_layer = conv_layer

                    self.conv_layers.append(conv_layer)
                    temp.append(conv_layer)

                
                self.conv_blocks.append(temp)
                
            else:
                for j in range(conv_block[i]):
                    if j == 0:
                        if j == conv_block[i] - 1:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_layer.max_pool_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=True, weight_init=weight_init)
                        else:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_layer.max_pool_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=False, weight_init=weight_init)
                            prev_conv_layer = conv_layer
                    else:
                        if j == conv_block[i] - 1:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_conv_layer.activation_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=True, weight_init=weight_init)
                        else:
                            conv_layer = VGG_Layer_Torch(kernels[i], activation_func[i], prev_conv_layer.activation_images.shape, batch_size, layer_num=j, stride=strides[i], last_layer_in_block=False, weight_init=weight_init)
                            prev_conv_layer = conv_layer

                    self.conv_layers.append(conv_layer)
                    temp.append(conv_layer)
                
                self.conv_blocks.append(temp)
                
            prev_layer = conv_layer

    def __iter__(self):
        return iter(self.conv_layers)
    
    def __len__(self):
        return len(self.conv_layers)
    
    def __getitem__(self, index):
        return self.conv_layers[index]
    
    def getLayer(self, index):
        return self.conv_layers[index]

    def getLastLayer(self):
        return self.conv_layers[-1]
    
    def size(self):
        return len(self.conv_layers)