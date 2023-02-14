
from abc import ABC, abstractmethod

class Model(ABC):
    
    def __init__(self, ModelHyperparams):
        ...
        pass
    
    @abstractmethod
    def train(X, y):
        pass
    
    @abstractmethod
    def predict_proba():
        pass
    