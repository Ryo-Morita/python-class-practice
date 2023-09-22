import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# fitとtransformを持つクラスを作る(中身は適当)
class MySelector(BaseEstimator, TransformerMixin):
    def __init__(self, use_cols):
        self.use_cols = use_cols

    def fit(self, X: pd.DataFrame)->None:
        return self
    
    def transform(self, X: pd.DataFrame)->pd.DataFrame:
        return X[self.use_cols]

# fitとtransformを持つクラスを作る(中身は適当)
class MyLogScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        return self
    
    def transform(self, X: pd.DataFrame):
        return np.log1p(X)
    
def main():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    num_train = 10000
    train_X = X.iloc[:num_train, :].reset_index(drop=True)
    test_X = X.iloc[num_train:, :].reset_index(drop=True)
    train_y = y[:num_train].values
    test_y = y[num_train:].values

    logscaler_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["MedInc", "Population"])),
        ("log_scaler", MyLogScaler())
    ])

    standardscaler_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["HouseAge", "AveRooms", "AveBedrms", "AveOccup"])),
        ("log_scaler", StandardScaler())
    ])

    raw_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["Latitude", "Longitude"]))
    ])

    feat_pipiline = FeatureUnion(transformer_list=[
        ("logscaler_pipeline", logscaler_pipeline),
        ("standardscaler_pipeline", standardscaler_pipeline),
        ("raw_pipeline", raw_pipeline)
        ])
    
    train_X = feat_pipiline.fit_transform(train_X)
    test_X = feat_pipiline.transform(test_X)
    assert train_X.shape[1] == test_X.shape[1]

    # Linear
    print("======LinearModel======")
    model = LinearRegression()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    r2 = r2_score(test_y, pred_y)
    print(f"RMSE : {np.round(rmse,3)}")
    print(f"R2 : {np.round(r2,3)}")

    print("======RandomForest======")
    model = RandomForestRegressor(random_state=1234)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    r2 = r2_score(test_y, pred_y)
    print(f"RMSE : {np.round(rmse,3)}")
    print(f"R2 : {np.round(r2,3)}")

if __name__ == "__main__":
    main()