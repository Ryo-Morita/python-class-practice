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

    def fit(self, X: pd.DataFrame, y=None)->None:
        return self
    
    def transform(self, X: pd.DataFrame,y=None)->pd.DataFrame:
        return X[self.use_cols]

# fitとtransformを持つクラスを作る(中身は適当)
class MyLogScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        return np.log1p(X)
    
def get_data():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    num_train = 10000
    train_X = X.iloc[:num_train, :].reset_index(drop=True)
    test_X = X.iloc[num_train:, :].reset_index(drop=True)
    train_y = y[:num_train].values
    test_y = y[num_train:].values
    return train_X, test_X, train_y, test_y

def print_score(test_y,pred_y):
    rmse = np.sqrt(mean_squared_error(test_y,pred_y))
    r2 = r2_score(test_y, pred_y)
    print(f"RMSE : {np.round(rmse,3)}")
    print(f"R2 : {np.round(r2,3)}")

    
def main():
    # データ取得
    train_X, test_X, train_y, test_y = get_data()

    # log変換器
    logscaler_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["MedInc", "Population"])),
        ("log_scaler", MyLogScaler())
    ])

    # 標準化変換器
    standardscaler_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["HouseAge", "AveRooms", "AveBedrms", "AveOccup"])),
        ("log_scaler", StandardScaler())
    ])

    # 何もしない変換器
    raw_pipeline = Pipeline([
        ('selector', MySelector(use_cols = ["Latitude", "Longitude"]))
    ])

    # 上記3変換器を結合した特徴量変換器
    feat_pipiline = FeatureUnion(transformer_list=[
        ("logscaler_pipeline", logscaler_pipeline),
        ("standardscaler_pipeline", standardscaler_pipeline),
        ("raw_pipeline", raw_pipeline)
        ])
    
    # 特徴量変換器とモデルの結合パイプライン(線形モデル)
    linearmodel_pipeline = Pipeline([
        ("feat_pipiline", feat_pipiline),
        ("LinearRegression",LinearRegression())
    ])

    # 特徴量変換器とモデルの結合パイプライン(randomoforest)
    rfmodel_pipeline = Pipeline([
        ("feat_pipiline", feat_pipiline),
        ("LinearRegression",RandomForestRegressor(random_state=1234))
    ])

    # 線形モデル実行
    print("======LinearModel======")
    linearmodel_pipeline.fit(train_X, train_y)
    pred_y = linearmodel_pipeline.predict(test_X)
    print_score(test_y, pred_y)

    # ランダムフォレスト実行
    print("======RandomForest======")
    rfmodel_pipeline.fit(train_X, train_y)
    pred_y = rfmodel_pipeline.predict(test_X)
    print_score(test_y, pred_y)

if __name__ == "__main__":
    main()