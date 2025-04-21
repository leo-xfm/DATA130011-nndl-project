from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self) -> None:
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, step_size=30, gamma=0.1) -> None:
        super().__init__()
        self.step_size = step_size
        self.gamma = gamma

    def step(self, optimizer) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, milestones=[800, 2400, 4000], gamma=0.5) -> None:
        super().__init__()
        self.milestones = milestones
        self.gamma = gamma
        self.idx = 0
    
    def step(self, optimizer) -> None:
        self.step_count += 1
        if self.idx < len(self.milestones) and self.milestones[self.idx] <= self.step_count:
            optimizer.init_lr *= self.gamma
            self.idx += 1

class MultiStepLR_layerWise(scheduler):
    def __init__(self, milestones=[800, 2400, 4000], gamma=0.5) -> None:
        super().__init__()
        self.milestones = milestones
        self.gamma = gamma
        self.idx = 0
    
    def step(self, optimizer) -> None:
        self.step_count += 1
        if self.idx < len(self.milestones) and self.step_count >= self.milestones[self.idx]:
            for layer_idx, lr in optimizer.layer_lrs.items():
                optimizer.layer_lrs[layer_idx] *= self.gamma                
            self.idx += 1
            
class ExponentialLR(scheduler):
    def __init__(self, gamma=0.9, last_epoch=-1) -> None:
        super().__init__()
        self.gamma = gamma
    
    def step(self, optimizer) -> None:
        self.step_count += 1
        optimizer.init_lr *= self.gamma
