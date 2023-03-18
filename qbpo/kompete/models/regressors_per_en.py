from typing import List, Union 
from sklearn import base as sk_base
from sklearn import linear_model as sk_lm

import numpy as np

import pandas as pd

COL_EN = "cfips"
COL_TS = "first_day_of_month"
COL_RTS = "running_ts"
Y = "y"

class LastKnownRegressorPerEn(sk_base.BaseEstimator, sk_base.RegressorMixin):
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

 

class LinearRegressorPerEn(sk_base.BaseEstimator, sk_base.RegressorMixin):
    def __init__(self, col_en:str = COL_EN, selected_columns:Union[str, List[str]] = None):
        self.col_en = col_en
        self.selected_columns = selected_columns 

    def _get_selected_columns(self, X):
        selected_columns = self.selected_columns if self.selected_columns is not None else X.columns
        if isinstance(selected_columns, str): selected_columns = [selected_columns]
        return selected_columns

    def fit(self, X, y):
        selected_columns = self._get_selected_columns(X)
        en_ts = [self.col_en] + selected_columns
        df = pd.DataFrame(X[en_ts])
        df[Y] = y
        
        self.models = {}
        for en, df_ in df.groupby(self.col_en):
            lm = sk_lm.LinearRegression()
            lm.fit(df_[selected_columns], df_[Y])
            self.models[en] = lm
        
        return self

    def predict(self, X):
        selected_columns = self._get_selected_columns(X)
        def _predict(en, values_of_selected_columns):
            lm = self.models.get(en, None)
            if lm:
                return lm.predict(np.array([values_of_selected_columns]))[0]
        return X.apply(lambda x : _predict(x[self.col_en], x[selected_columns]), axis=1)
        


