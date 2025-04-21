from abc import abstractmethod
import numpy as np
from . import op

class Module:
    def __init__(self, layers=None, outputs=None):
        self.layers = layers if layers else []
        self.outputs = None
        self.print_shapes = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):        
        outputs = X
        if self.print_shapes:
            print(outputs.shape)
        for layer in self.layers:
            outputs = layer(outputs)
            if self.print_shapes:
                print(outputs.shape)
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        if self.print_shapes:
            print(grads.shape)
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
            if self.print_shapes:
                print(grads.shape)
        return grads
    
class MLP(Module):
    def __init__(self, size_list, act_func=None, lambda_list=None, dropout_rate=None, batch_norm=False):
        super().__init__()
        
        self.size_list = size_list
        is_decay = lambda_list is not None
        
        if act_func is None:
            raise ValueError(f"Activation function is None.")
        elif size_list is not None:
            self.layers = []
            
            if batch_norm:
                self.layers.append(op.BatchNorm1D(size_list[0]))
            
            if lambda_list is not None and len(lambda_list) != len(size_list) - 1:
                raise ValueError(f"Length of lambda_list must be {len(size_list) - 1}, but got {len(lambda_list)}.")

            for i in range(len(size_list) - 1):
                self.layers.append ( op.Linear(size_list[i], 
                                            size_list[i + 1], 
                                            is_decay, 
                                            lambda_list[i] if lambda_list is not None else None) 
                                    )
                if i < len(size_list) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(op.ReLU())
                    elif act_func == 'Sigmoid':
                        self.layers.append(op.Sigmoid())                    
                    if dropout_rate:
                        self.layers.append(op.Dropout(dropout_rate))
                        
        self.layers.append(op.Softmax())

class CNN(Module):
    def __init__(self):
        super().__init__()
        self.print_shapes = False
        self.layers = [ # (4096, 1, 28, 28)
            op.conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, weight_decay=True),  # (4096, 8, 14, 14)
            op.Sigmoid(),
            op.MaxPool2D(pool_size=(2, 2), stride=2), # (4096, 8, 7, 7)
            op.Flatten(), # (4096, 392)
            op.Linear(in_dim=392, out_dim=10, weight_decay=True), # (4096, 10)
            op.Softmax()
        ]

class CNN_best(Module):
    def __init__(self):
        super().__init__()
        self.print_shapes = False
        self.layers = [ # (4096, 1, 28, 28)
            op.conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, weight_decay=True),  # (4096, 32, 26, 26)
            op.ReLU(),
            op.Dropout(0.25),
            op.BatchNorm2D(32),
            op.MaxPool2D(pool_size=(2, 2), stride=2), # (4096, 32, 13, 13)
            
            op.conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, weight_decay=True),  # (4096, 64, 11, 11)
            op.ReLU(),
            op.BatchNorm2D(64),
            op.MaxPool2D(pool_size=(2, 2), stride=2), # (4096, 64, 5, 5)
            
            op.conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, weight_decay=True),  # (4096, 64, 3, 3)
            op.ReLU(),       
                        
            op.Flatten(), # (4096, 576)
            op.Linear(in_dim=576, out_dim=256, weight_decay=True), # (4096, 256)
            op.ReLU(),
            
            op.Linear(in_dim=256, out_dim=10, weight_decay=True),
            op.Dropout(0.5),
            op.Softmax()
        ]

class CNN_best_simplified(Module):
    def __init__(self):
        super().__init__()
        self.print_shapes = False
        self.layers = [ # (4096, 1, 28, 28)
            op.conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=5, padding=0, weight_decay=True),  # (4096, 64, 6, 6)
            op.ReLU(),
            # op.Dropout(0.25),
            # op.BatchNorm2D(64),
            op.MaxPool2D(pool_size=(2, 2), stride=2), # (4096, 64, 3, 3)
                        
            op.Flatten(), # (4096, 576)
            op.Linear(in_dim=576, out_dim=256, weight_decay=True), # (4096, 256)
            op.ReLU(),
            
            op.Linear(in_dim=256, out_dim=10, weight_decay=True), # (4096, 10)
            # op.Dropout(0.5),
            op.Softmax()
        ]
        
class LinearResNet(Module):
    def __init__(self, num_classes=10, in_dim=784, weight_decay=True, weight_decay_lambda=1e-4):
        super().__init__()
        
        self.layers = [
            op.Linear(in_dim=784, out_dim=98, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda), # 0
            op.Sigmoid(),
            op.Linear(in_dim=98, out_dim=25, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda), # 2
            op.Sigmoid(),
            op.Linear(in_dim=25, out_dim=98, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda), # 4
            op.Sigmoid(),
            op.Linear(in_dim=98, out_dim=784, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda), # 6
            op.Sigmoid(),
            op.Linear(in_dim=784, out_dim=10, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda), # 8
            op.Sigmoid(),
            op.Softmax()
        ]
    
    def forward(self, X):
        X1 = self.layers[0](X) # 98
        X1 = self.layers[1](X1)
        
        X2 = self.layers[2](X1) # 25
        X2 = self.layers[3](X2)
        
        X3 = self.layers[4](X2) # 98
        X3 = self.layers[5](X3)
        X3 = X3 + X1  # Residual connection
        
        X4 = self.layers[6](X3) # 784
        X4 = self.layers[7](X4)
        X4 = X4 + X  # Residual connection

        output = self.layers[8](X4)
        output = self.layers[9](output)
        
        return output

