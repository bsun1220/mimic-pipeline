from abc import ABC, abstractmethod

class Disease(ABC):
    
    def __init__(self):
        ...
        pass
    
    
    
    @abstractmethod
    def predict_proba():
        pass
    
    