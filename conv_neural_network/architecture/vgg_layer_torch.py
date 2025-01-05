import torch
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print("Device: ",device)
# torch.set_default_device(device)


class VGG_Layer_Torch():
    def __init__(self, kernels, activation_func, image_shape, batch_size, layer_num, stride=1, last_layer_in_block=False, weight_init=None):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.kernels = torch.empty(kernels[0], kernels[1], kernels[1], image_shape[3])
        if weight_init == "xavier":
            torch.nn.init.xavier_normal_(self.kernels)
        elif weight_init == "xavier-uniform":
            torch.nn.init.xavier_uniform_(self.kernels)
        elif weight_init == "kaiming-out":
            torch.nn.init.kaiming_normal_(self.kernels, mode='fan_out', nonlinearity='relu', a=0)
        elif weight_init == "kaiming-in":
            torch.nn.init.kaiming_normal_(self.kernels, mode='fan_in', nonlinearity='relu', a=0)
        elif weight_init == "kaiming-uniform":
            torch.nn.init.kaiming_uniform_(self.kernels, a=0)
        else:
            torch.nn.init.kaiming_normal_(self.kernels, mode='fan_out', nonlinearity='relu', a=0)

        self.activation_func = activation_func
        self.last_layer_in_block = last_layer_in_block
        self.activation_images = torch.zeros((
            batch_size,
            image_shape[1] - self.kernels.shape[1] + 1,
            image_shape[2] - self.kernels.shape[2] + 1,
            self.kernels.shape[0]
        ))
    
        if last_layer_in_block:
            self.max_pool_images = torch.zeros((
                batch_size,
                self.activation_images.shape[1] // 2,
                self.activation_images.shape[2] // 2,
                self.activation_images.shape[3]
            ))
        else:
            self.max_pool_images = torch.tensor([])
        self.stride = stride
        