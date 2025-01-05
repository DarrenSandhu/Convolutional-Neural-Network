import torch
from architecture.fully_connected_layer_torch import Fully_Connected_Layer

class Fully_Connected_Layers_Torch():
    def __init__(self, layers_number, output_sizes, activation_funcs, weight_init=None):
        if len(output_sizes) != layers_number:
            raise ValueError("The number of output sizes must match the number of layers")
        if len(activation_funcs) != layers_number:
            raise ValueError("The number of activation functions must match the number of layers")
        
        self.fc_layers = []
        prev_layer_size = output_sizes[0]
        for i in range(1,layers_number,1):
            fc_layer = Fully_Connected_Layer(prev_layer_size, output_sizes[i], activation_funcs[i], weight_init)
            prev_layer_size = output_sizes[i]
            self.fc_layers.append(fc_layer)
    
    def __iter__(self):
        return iter(self.fc_layers)
    
    def __len__(self):
        return len(self.fc_layers)
    
    def __getitem__(self, index):
        return self.fc_layers[index]
    
    def getLayer(self, index):
        return self.fc_layers[index]

    def getLastLayer(self):
        return self.fc_layers[-1]
    
    def size(self):
        return len(self.fc_layers)