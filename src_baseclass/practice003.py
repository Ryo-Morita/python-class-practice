from abc import ABC, abstractmethod
import numpy as np

class Base(ABC):    
    @abstractmethod
    def fit(self, X):
        raise NotImplementedError()
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def hoge(self):
        print("hoge")
    
class Calc(Base):
    def __init__(self):
        self.mu_ = None

    def fit(self,X):
        self.mu_ = np.mean(X)
        return self

    def transform(self, X):
        out = X - self.mu_
        return out

#############
x = np.array([1,2,3,4,5,6])

calc = Calc() 
calc.fit(x)
out = calc.transform(x)
print(out) # [-2.5 -1.5 -0.5  0.5  1.5  2.5]

#############
x = np.array([1,2,3,4,5,6])

calc = Calc() 
out = calc.fit_transform(x)
print(out) # [-2.5 -1.5 -0.5  0.5  1.5  2.5]
calc.hoge()

#############
train_x = np.array([1,2,3,4,5,6])
test_x = np.array([2,3,4,5,6,7,8,9])

calc = Calc() 
train_x = calc.fit_transform(train_x)
test_x = calc.transform(test_x)
print(train_x) # [-2.5 -1.5 -0.5  0.5  1.5  2.5]
print(test_x) # [-1.5 -0.5  0.5  1.5  2.5  3.5  4.5  5.5]