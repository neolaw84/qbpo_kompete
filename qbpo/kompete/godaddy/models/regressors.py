import abc

from sklearn import base as sk_base
from sklearn import linear_model as sk_lm

import pandas as pd

COL_EN = "cfips"
COL_TS = "first_day_of_month"
Y = "y"

class LastKnownRegressor(sk_base.BaseEstimator, sk_base.RegressorMixin):
    def __init__(self, col_en:str = COL_EN, col_ts:str = COL_TS, return_df:bool = False):
        self.col_en = col_en
        self.col_ts = col_ts
        self.return_df = return_df

    def fit(self, X, y):
        en_ts = [self.col_en, self.col_ts]
        df = pd.DataFrame(X[en_ts])
        df[Y] = y
        df_max = X[en_ts].groupby(by=self.col_en).max().reset_index()
        self.df_y = pd.merge(df_max, df, how="inner", on=en_ts)
        self.df_y.drop(self.col_ts, axis=1, inplace=True)
        self.df_y.set_index(self.col_en, inplace=True)
        return self

    def predict(self, X):
        if self.return_df:
            return X.apply(lambda x : self.df_y.loc[x[self.col_en], Y], axis=1)
        else:
            return X.apply(lambda x : self.df_y.loc[x[self.col_en], Y].item(), axis=1)

COL_TARGET = "active"
 

class LinearRegressor(sk_base.BaseEstimator, sk_base.RegressorMixin):
    def __init__(self, col_en:str = COL_EN, col_running_ts:str = "running_ts"):
        self.col_en = col_en
        self.col_running_ts = col_running_ts

    def fit(self, X, y):
        en_ts = [self.col_en, self.col_running_ts]
        df = pd.DataFrame(X[en_ts])
        df["y"] = y
        
        self.models = {}
        for en, df_ in df.groupby(self.col_en):
            lm = sk_lm.LinearRegression()
            lm.fit(df_[[self.col_running_ts]], df_["y"])
            self.models[en] = lm
        
        return self

    def predict(self, X):
        def _predict(en, running_ts):
            lm = self.models.get(en, None)
            if lm:
                return lm.predict(pd.DataFrame({self.col_running_ts: [running_ts]}))
        return X.apply(lambda x : _predict(x[self.col_en], x[self.col_running_ts]), axis=1)
        


