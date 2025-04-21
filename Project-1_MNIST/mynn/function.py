import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def __call__(self, outputs, y_true):
        m = outputs.shape[0]
        one_hot_y = np.zeros((m, outputs.shape[1]))
        one_hot_y[range(m), y_true] = 1
        outputs = outputs.reshape(m, -1)
        loss = -np.sum(one_hot_y * np.log(outputs)) / m
        return loss
    
    def backward(self, outputs, y_true):
        m = outputs.shape[0]
        one_hot_y = np.zeros((m, outputs.shape[1]))
        one_hot_y[range(m), y_true] = 1
        outputs = outputs.reshape(m, -1)
        return (outputs - one_hot_y) / m