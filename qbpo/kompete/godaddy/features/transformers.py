from warnings import warn
from typing import Union, List

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from sklearn import base as sk_base
from sklearn import pipeline as sk_pipe

from qbpo.utils import feature as feat_utils
import qbpo.config


COL_ROW_ID = "row_id"
COL_TS = "first_day_of_month"
COL_EN = "cfips"
COL_COUNTY = "county"
COL_STATE = "state"
COL_TARGET = "microbusiness_density"
COL_ACTIVE = "active"


def merge_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
    assert (
        df_train.cfips.nunique() * df_train.first_day_of_month.nunique()
        == df_train.shape[0]
    )
    assert (
        df_test.cfips.nunique() * df_test.first_day_of_month.nunique()
        == df_test.shape[0]
    )

    df_cfips_info = (
        df_train[[COL_EN, COL_COUNTY, COL_STATE]]
        .drop_duplicates(inplace=False)
        .set_index(COL_EN)
    )
    df_test[COL_COUNTY] = df_test.apply(
        lambda x: df_cfips_info.loc[x.cfips, COL_COUNTY], axis=1
    )
    df_test[COL_STATE] = df_test.apply(
        lambda x: df_cfips_info.loc[x.cfips, COL_STATE], axis=1
    )

    df_all = pd.merge(
        df_train,
        df_test,
        how="outer",
        on=[COL_ROW_ID, COL_TS, COL_EN, COL_COUNTY, COL_STATE],
        sort=False,
    )
    assert df_train.shape[0] + df_test.shape[0] == df_all.shape[0]

    df_all.sort_values(by=COL_TS, inplace=True)
    df_all.set_index(COL_ROW_ID, inplace=True)
    return df_all


def _set_default(value, default):
    return default if not value else value


class FeatureNamesOutMixin:
    def get_feature_names_out(self, input_features: List[str] = None):
        if input_features is None:
            input_features = self.columns

        output_columns = []

        for args in self.out_columns_args:
            output_columns.extend(
                feat_utils.generate_output_columns(*args, columns=input_features)
            )
        return output_columns


class SortByIndexMixin:
    def sort_by_index(self, df: pd.DataFrame, inplace: bool = False):
        if self.index is None:
            warn("index is None, not sorting, we quit")
            return df
        val_to_idx = {val: idx for idx, val in enumerate(self.index.values)}

        def _key(x):
            return [val_to_idx.get(item) for item in x]

        temp_df = df.sort_index(key=_key, inplace=inplace)
        return None if inplace else temp_df


class ShiftFeatures(sk_base.TransformerMixin, FeatureNamesOutMixin, SortByIndexMixin):
    def __init__(
        self,
        period_from: int = 1,
        period_to: int = 2,
        columns: Union[str, List[str]] = None,
        id_columns: Union[str, List[str]] = None,
        index: pd.Index = None,
    ):
        self.period_from = period_from
        self.period_to = period_to
        self.columns = columns
        self.id_columns = id_columns
        self.index = index
        self.out_columns_args = [["shift", p] for p in range(period_from, period_to)]

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.columns = _set_default(
            self.columns,
            [
                c
                for c in X.columns
                if c not in [COL_EN, COL_ACTIVE]
                and c in X.select_dtypes(include=np.number).columns
            ],
        )

        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        columns = self.columns

        id_columns = _set_default(self.id_columns, [COL_EN])

        output_columns = self.get_feature_names_out()

        df = pd.DataFrame(data=None, index=X.index)
        for period in tqdm(
            range(self.period_from, self.period_to),
            disable=qbpo.config.tqdm_config["disable"],
        ):
            df_ = feat_utils.shift(
                X,
                columns=columns,
                period=period,
                id_columns=id_columns,
                output_columns=output_columns,
            )

            df = pd.merge(
                df, df_, left_index=True, right_index=True, suffixes=["", "__to_drop"]
            )
            df = df[[c for c in df.columns if not c.endswith("__to_drop")]]

        if self.index is not None:
            self.sort_by_index(df, inplace=True)
        print(df.shape)
        return df


class DiffFeatures(sk_base.TransformerMixin, FeatureNamesOutMixin, SortByIndexMixin):
    def __init__(
        self,
        period_from: int = 1,
        period_to: int = 2,
        columns: Union[str, List[str]] = None,
        id_columns: Union[str, List[str]] = None,
        index: pd.Index = None,
    ):
        self.period_from = period_from
        self.period_to = period_to
        self.columns = columns
        self.id_columns = id_columns
        self.index = index
        self.out_columns_args = [["diff", p] for p in range(period_from, period_to)]

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.columns = _set_default(
            self.columns,
            [
                c
                for c in X.columns
                if c not in [COL_EN, COL_ACTIVE]
                and c in X.select_dtypes(include=np.number).columns
            ],
        )
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        columns = self.columns

        id_columns = _set_default(self.id_columns, [COL_EN])

        df = pd.DataFrame(data=None, index=X.index)
        for period in tqdm(
            range(self.period_from, self.period_to),
            disable=qbpo.config.tqdm_config["disable"],
        ):
            df_ = feat_utils.diff(
                X,
                columns=columns,
                period=period,
                id_columns=id_columns,
                output_columns=self.get_feature_names_out(),
            )
            df = pd.merge(df, df_, left_index=True, right_index=True)

        if self.index is not None:
            self.sort_by_index(df, inplace=True)

        return df


class RollingFeatures(sk_base.TransformerMixin, FeatureNamesOutMixin, SortByIndexMixin):
    def __init__(
        self,
        period: int = 3,
        columns: Union[str, List[str]] = None,
        id_columns: Union[str, List[str]] = None,
        funcs=[np.mean, np.sum, np.std],
        index: pd.Index = None,
    ):
        self.period = period
        self.columns = columns
        self.id_columns = id_columns
        self.funcs = funcs
        self.out_columns_args = [["roll", period]]
        self.index = index

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.columns = _set_default(
            self.columns,
            [
                c
                for c in X.columns
                if c not in [COL_EN, COL_ACTIVE]
                and c in X.select_dtypes(include=np.number).columns
            ],
        )
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        columns = self.columns

        id_columns = _set_default(self.id_columns, [COL_EN])

        df = pd.DataFrame(data=None, index=X.index)
        for func in tqdm(self.funcs, disable=config.tqdm_config["disable"]):
            df_ = feat_utils.rolling(
                X,
                columns=columns,
                period=self.period,
                id_columns=id_columns,
                agg_func=func,
                output_columns=self.get_feature_names_out(),
            )
            df = pd.merge(df, df_, left_index=True, right_index=True)

        if self.index is not None:
            self.sort_by_index(df, inplace=True)

        return df


class PassThroughFeatures(
    sk_base.TransformerMixin, FeatureNamesOutMixin, SortByIndexMixin
):
    def __init__(self, copy=True, index: pd.Index = None, columns: List[str] = None):
        self.copy = copy
        self.out_columns_args = [[]]
        self.index = index
        self.columns = columns

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        if self.columns is None and isinstance(X, pd.DataFrame):
            self.columns = list(X.columns.values)
        return self

    def transform(self, X):
        df = X.copy() if self.copy else X
        if self.index is not None:
            self.sort_by_index(df, inplace=True)
        return df


class MakeDataFrame(sk_base.TransformerMixin):
    def __init__(self, index: pd.Index, columns: List[str], infer_objects: bool = True):
        self.index = index
        self.columns = columns
        self.infer_objects = infer_objects

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = pd.DataFrame(data=X, columns=self.columns, index=self.index)
        if self.infer_objects:
            df = df.infer_objects()
        return df
