from abc import abstractmethod
import numpy as np

class Optimizer:
    def __init__(self, init_lr) -> None:
        self.init_lr = init_lr

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr):
        super().__init__(init_lr)
    
    def step(self, model):
        for layer in model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]

class MomentGD(Optimizer):
    def __init__(self, init_lr, mu):
        super().__init__(init_lr)
        self.mu = mu
        self.velocities = {}
    
    def step(self, model):
        for layer_idx, layer in enumerate(model.layers):
            if layer.optimizable:
                if layer_idx not in self.velocities:
                    self.velocities[layer_idx] = {}
                for key in layer.params.keys():
                    if key not in self.velocities[layer_idx]:
                        self.velocities[layer_idx][key] = np.zeros_like(layer.params[key])
                    if layer.grads[key].shape != self.velocities[layer_idx][key].shape:
                        raise ValueError(f"Gradient and momentum shapes do not match: {layer.grads[key].shape} vs {self.velocities[layer_idx][key].shape}")
                    self.velocities[layer_idx][key] = self.mu * self.velocities[layer_idx][key] + self.init_lr * layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.velocities[layer_idx][key]

class SGD_layerWise(Optimizer):
    def __init__(self, init_lr, layer_lrs=None):
        super().__init__(init_lr)
        self.layer_lrs = layer_lrs if layer_lrs else {}

    def step(self, model):
        for layer_idx, layer in enumerate(model.layers):
            if layer.optimizable == True:
                lr = self.layer_lrs.get(layer_idx, self.init_lr) 
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]