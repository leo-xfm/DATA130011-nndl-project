from abc import abstractmethod
import numpy as np
from . import function
import pickle

class Model:
    def __init__(self, module=None, loss_fn=None, optimizer=None, scheduler=None):
        self.module = module
        self.loss_fn = getattr(function, loss_fn)()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training = True
    
    def __call__(self, X):
        return self.module(X)
    
    def forward(self, X):
        return self.module(X)
    
    def backward(self, output, y):
        loss = self.loss_fn.backward(output, y)
        return self.module.backward(loss)
    
    def zero_grad(self):
        for layer in self.module.layers:
            if layer.optimizable:
                for key in layer.grads:
                    layer.grads[key] = None
    
    def optimizerStep(self):
        self.optimizer.step(self.module)
    
    def schedulerStep(self):
        self.scheduler.step(self.optimizer)
    
    def train(self):
        self.training = True
        for layer in self.module.layers:
            layer.training = True
    
    def eval(self):
        self.training = False
        for layer in self.module.layers:
            layer.training = False
    
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            model_data = {
                'model': self.module,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'loss_fn': self.loss_fn,
                'training': self.training
            }
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")
        
    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            model_data = pickle.load(f)
            self.module = model_data['model']
            self.optimizer = model_data['optimizer']
            self.scheduler = model_data['scheduler']
            self.loss_fn = model_data['loss_fn']
            self.training = model_data['training']
        print(f"Model loaded from {save_path}")