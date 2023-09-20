class Base:
    def fit(self, X):
        raise NotImplementedError()
    
    def predict(self, X):
        raise NotImplementedError()
    
class Calc(Base):
    def __init__(self):
        pass

calc = Calc() #Errorは出ない
