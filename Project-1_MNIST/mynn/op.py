from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self, training = True) -> None:
        self.optimizable = True
        self.training = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass 
    
class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, weight_decay=False, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros((1,out_dim))
        self.grads = {'W':None, 'b':None}
        self.input = None # Record the input for backward process.
        
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
    
    def __call__(self,X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        bsz = self.input.shape[0]
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input<0, 0, grads)
        return output
    
    def clear_grad(self):
        pass
    
class Sigmoid(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        return 1/(1+np.exp(-X))

    def backward(self, grads):
        return grads*(1-grads)
    
    def clear_grad(self):
        pass

class Softmax(Layer):
    """
    An output layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, grads):
        return grads
    
    def clear_grad(self):
        pass

class Dropout(Layer):
    def __init__(self, drop_rate=0.5) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.input = None
        self.training = True
        self.optimizable = False
        
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        if self.training:
            self.mask = np.random.rand(*X.shape) > self.drop_rate
            output = X * self.mask / (1 - self.drop_rate)
            return output
        else:
            return X
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        return grads * self.mask

    def clear_grad(self):
        pass
    
class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.randn, weight_decay=False, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        self.params = {
            'W': initialize_method(out_channels, in_channels, kernel_size, kernel_size)*0.1,
            'b': np.zeros((out_channels, 1))
        }
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.X_padded = None
        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        no padding
        """
        self.input = X
        # print("forward.X", X.shape) # (4096, 1, 28, 28)
        bsz, in_channels, H, W = X.shape
        H_out = (H - self.kernel_size + 2*self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2*self.padding) // self.stride + 1
        output = np.zeros((bsz, self.out_channels, H_out, W_out))
        # print("forward.output", output.shape) # (4096, 8, 28, 28)
        
        if self.padding > 0:
            X_padded = np.pad(X, 
                              ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant')
        else:
            X_padded = X
        
        self.X_padded = X_padded
        
        for i in range(H_out):
            for j in range(W_out):
                H1 = i * self.stride
                H2 = H1 + self.kernel_size
                W1 = j * self.stride
                W2 = W1 + self.kernel_size
                
                x_slice = X_padded[:, :, H1:H2, W1:W2]
                
                for k in range(self.out_channels):
                    # print(output[:, k, i, j].shape, x_slice.shape, self.params['W'][k].shape, self.params['b'][k])
                    output[:, k, i, j] = np.sum(x_slice * self.params['W'][k, :, :, :], axis=(1, 2, 3)) + self.params['b'][k]
        
        return output
        

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        bsz, out_channels, H_out, W_out = grads.shape
        in_channels, H, W = self.input.shape[1:] # (4096, 1, 28, 28)
        # print(grads.shape, bsz, out_channels, H_out, W_out) # (4096, 8, 14, 14)
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        grad_input = np.zeros_like(self.input)
        
        X = self.input
        
        for i in range(H_out-self.padding):
            for j in range(W_out-self.padding):
                H1 = i * self.stride
                H2 = H1 + self.kernel_size
                W1 = j * self.stride
                W2 = W1 + self.kernel_size
                
                x_slice = X[:, :, H1:H2, W1:W2]
                
                for k in range(out_channels):
                    # print(i, j, k, grad_input[:, :, H1:H2, W1:W2].shape, self.params['W'][k].shape, grads[:, k, i, j][:, None, None, None].shape)
                    self.grads['b'][k] += np.sum(grads[:, k, i, j], axis=0, keepdims=True)
                    self.grads['W'][k] += np.sum(x_slice * grads[:, k, i, j][:, None, None, None], axis=0)
                    grad_input[:, :, H1:H2, W1:W2] += self.params['W'][k] * grads[:, k, i, j][:, None, None, None]
                    
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
        
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class MaxPool2D(Layer):
    def __init__(self, pool_size=(2,2), stride=1, padding=0):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.X_padded = None
        self.optimizable = False
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        # print("forward.X", X.shape)
        bsz, in_channels, H, W = X.shape
        pool_height, pool_width = self.pool_size
        H_out = (H - pool_height + 2*self.padding) // self.stride + 1
        W_out = (W - pool_width + 2*self.padding) // self.stride + 1
        output = np.zeros((bsz, in_channels, H_out, W_out))
        # print("forward.output", output.shape)
        
        if self.padding > 0:
            X_padded = np.pad(X, 
                              ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant')
        else:
            X_padded = X
        
        self.X_padded = X_padded
        
        for i in range(H_out):
            for j in range(W_out):
                x_slice = X_padded[:, 
                                   :, 
                                   (i*self.stride) : (i*self.stride+pool_height), 
                                   (j*self.stride) : (j*self.stride+pool_width)]
                output[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        
        # print(output.shape) (4096, 8, 14, 14)
        return output

    def backward(self, grads):
        batch_size, in_channels, H_out, W_out = grads.shape
        pool_height, pool_width = self.pool_size
        grad_input = np.zeros_like(self.input)
        
        X_padded = self.X_padded
        
        for i in range(H_out):
            for j in range(W_out):
                x_slice = X_padded[:, :, 
                                   (i*self.stride):(i*self.stride+pool_height), 
                                   (j*self.stride):(j*self.stride+pool_width)]
                max_vals = np.max(x_slice, axis=(2, 3))
                
                for b in range(batch_size):
                    for c in range(in_channels):
                        mask = (x_slice[b, c, :, :] == max_vals[b, c])
                        grad_input[b, c, 
                                   (i*self.stride):(i*self.stride+pool_height), 
                                   (j*self.stride):(j*self.stride+pool_width)] +=\
                                       grads[b, c, i, j] * mask

        return grad_input
        
    def clear_grad(self):
        pass

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], np.prod(X.shape[1:]))
    
    def backward(self, grads):
        return grads.reshape(self.input_shape)
    
    def clear_grad(self):
        pass

class BatchNorm2D(Layer):
    """
    2D Batch Normalization layer.
    This layer normalizes the input to have zero mean and unit variance.
    It also includes learnable parameters `gamma` and `beta` for scaling and shifting.
    """
    def __init__(self, num_features, momentum=0.9, eps=1e-5, weight_decay=False, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        self.params = {
            'gamma': np.ones((1, num_features, 1, 1)),  # [1, num_features, 1, 1]
            'beta': np.zeros((1, num_features, 1, 1))   # [1, num_features, 1, 1]
        }
        
        self.grads = {'gamma': None, 'beta': None}
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        self.input = None
        self.optimizable = True

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Normalize the input and apply batch normalization.
        X: [batch_size, num_features, H, W]
        """
        self.input = X
        batch_size, num_features, H, W = X.shape
        mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # [1, num_features, 1, 1]
        var = np.var(X, axis=(0, 2, 3), keepdims=True)  # [1, num_features, 1, 1]

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        X_norm = (X - mean) / np.sqrt(var + self.eps)  # [batch_size, num_features, H, W]
        output = self.params['gamma'] * X_norm + self.params['beta']
        return output

    def backward(self, grads):
        """
        grads: [batch_size, num_features, H, W]
        """
        batch_size, num_features, H_out, W_out = grads.shape
        mean = np.mean(self.input, axis=(0, 2, 3), keepdims=True)
        var = np.var(self.input, axis=(0, 2, 3), keepdims=True)
        X_norm = (self.input - mean) / np.sqrt(var + self.eps)

        grad_gamma = np.sum(grads * X_norm, axis=(0, 2, 3), keepdims=True)
        grad_beta = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        if self.weight_decay:
            grad_gamma += self.weight_decay_lambda * grad_gamma
            grad_beta += self.weight_decay_lambda * grad_beta

        grad_input = (grads - np.mean(grads, axis=(0, 2, 3), keepdims=True) - 
                      X_norm * np.mean(grads * X_norm, axis=(0, 2, 3), keepdims=True)) / np.sqrt(var + self.eps)

        self.grads = {'gamma': grad_gamma, 'beta': grad_beta}
            
        return grad_input

    def clear_grad(self):
        self.grads = {'gamma': None, 'beta': None}
