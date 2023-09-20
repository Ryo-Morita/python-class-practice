from abc import ABC, abstractmethod

class Base(ABC):    
    @abstractmethod
    def fit(self, X):
        raise NotImplementedError()
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()
    
    def fit_transform(self, X):
        return self.fit(X).transform()
    
    def hoge(self):
        print("hoge")
    
class Calc(Base):
    def __init__(self):
        pass

calc = Calc() # Error TypeError: Can't instantiate abstract class Calc with abstract methods fit, predict