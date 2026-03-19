from abc import ABC, abstractmethod

class BaseMLModel(ABC):
    
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, text: str):
        pass