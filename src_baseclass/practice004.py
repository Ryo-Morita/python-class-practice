import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Calc(BaseEstimator, TransformerMixin):
    def __init__(self, alpha = 10):
        self.mu_ = None
        self.alpha = alpha # 使用しないけどハイパラとして持っておく。これがget_params()で取れるようになる

    def fit(self,X):
        self.mu_ = np.mean(X)
        return self

    def transform(self, X):
        out = X - self.mu_
        return out

#############
train_x = np.array([1,2,3,4,5,6])
test_x = np.array([2,3,4,5,6,7,8,9])

calc = Calc() 
train_x = calc.fit_transform(train_x)
test_x = calc.transform(test_x)
print(train_x) # [-2.5 -1.5 -0.5  0.5  1.5  2.5]
print(test_x) # [-1.5 -0.5  0.5  1.5  2.5  3.5  4.5  5.5]

print(calc.get_params())