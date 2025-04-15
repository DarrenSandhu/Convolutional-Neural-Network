import torch
import math

class Fully_Connected_Layer():
    def __init__(self, input_layer, output_size, activation_func, weight_init=None):
        self.in_features = input_layer
        self.out_features = output_size
        self.weights = torch.empty(self.in_features, self.out_features)
        if activation_func == "sigmoid":
            torch.nn.init.xavier_normal_(self.weights)
        else:
            if weight_init == "xavier":
                torch.nn.init.xavier_normal_(self.weights)
            elif weight_init == "xavier-uniform":
                torch.nn.init.xavier_uniform_(self.weights)
            elif weight_init == "kaiming":
                torch.nn.init.kaiming_normal_(self.weights)
            elif weight_init == "kaiming-out":
                torch.nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu', a=0)
            elif weight_init == "kaiming-in":
                torch.nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu', a=0)
            elif weight_init == "kaiming-uniform":
                torch.nn.init.kaiming_uniform_(self.weights, a=0)
            else:
                torch.nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu', a=0)
        # print("Fully Connected Weights: ", self.weights.shape)
        # print("\n")

        # self.biases = torch.randn(1, self.out_features) * 0.01
        self.biases = torch.zeros(1, self.out_features)

        # self.shape = input_layer.shape

        self.activation_func = activation_func

        self.final_output = None
