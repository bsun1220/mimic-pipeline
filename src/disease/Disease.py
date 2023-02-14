'''
Author: Tony
Classes for each disease in order to find patients and relevant features
'''
from abc import ABC, abstractmethod
from loader import Loader

class Disease(ABC):
    
    def __init__(self):
        ...
        pass
    
    @abstractmethod
    def get_patients():
        """
        Obtain all patients with this particular disease
        """
        pass


class Sepsis(Disease):
    
    def __init__(self):
        super().__init__()
        pass
    

class HeartAttack(Disease):
    
    def __init__(self):
        super().__init__()
        pass


class HeartFailure(Disease):
    
    def __init__(self):
        super().__init__()
        pass
    
    
if __name__=="__main__":
    import sys
    print(sys.path)
    loader = Loader()
    
    