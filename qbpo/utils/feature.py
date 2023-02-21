import functools
import itertools
import inspect
from typing import Union, List

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import qbpo.config


def get_arg(func, arg_name, *args, **kwargs):
    argspec = inspect.getfullargspec(func)
    sorted_args = argspec.args
    if arg_name in sorted_args:
        idx = sorted_args.index(arg_name)
        if idx < len(args):
            arg_value = args[idx]
        elif arg_name in kwargs.keys():
            arg_value = kwargs[arg_name]
        else:
            arg_value = None
    return arg_value


def func_group_by_key(func=None, id_column_arg_name: str = "id_columns"):
    @functools.wraps(func)
    def wrapper_func_group_by_key(*args, **kwargs):
        id_columns = get_arg(func, id_column_arg_name, *args, **kwargs)
        df = get_arg(func, "df", *args, **kwargs)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        if id_columns:
            dfs = df.groupby(by=id_columns)
            _dfs = []
            for _, _df in tqdm(dfs, disable=qbpo.config.tqdm_config["disable"]):
                args = (a for a in args if a is not df)
                kwargs = {k: v for k, v in kwargs.items() if v is not df}
                temp_df = func(_df, *args, **kwargs)
                _dfs.append(temp_df)
            df_feature = pd.concat(_dfs)
        else:
            df_feature = func(*args, **kwargs)

        return df_feature

    return wrapper_func_group_by_key


def _ensure_list(str_param: Union[str, List[str]]):
    return [str_param] if isinstance(str_param, str) else str_param


def generate_output_columns(*args, columns: List[str] = []):
    column_template = "__".join(
        ["{{{}}}".format(idx) for idx in range(0, len(args) + 1)]
    )
    output_columns = [column_template.format(c, *args) for c in columns]
    return output_columns


def _shift(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    output_columns: Union[str, List[str]] = None,
):
    temp_df = pd.DataFrame(index=df.index)
    for c, oc in zip(columns, output_columns):
        temp_df[oc] = df[c].shift(periods=period)
    return temp_df


@func_group_by_key
def shift(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    id_columns: Union[str, List[str]] = None,
    output_columns: Union[str, List[str]] = None,
):
    columns = _ensure_list(columns)

    assert all([c in df.columns for c in columns])

    output_columns = (
        generate_output_columns("shift", period, columns=columns)
        if not output_columns
        else output_columns
    )

    df_shifted = _shift(
        df, columns=columns, period=period, output_columns=output_columns
    )

    return df_shifted


def diff_columns(
    df: pd.DataFrame, columns: Union[str, List[str]], mirror: bool = False
):
    df_diff = pd.DataFrame(index=df.index)
    pairs = itertools.product(columns, columns)
    for ka, kb in tqdm(pairs, disable=qbpo.config.tqdm_config["disable"]):
        if ka == kb:
            continue
        if not mirror and ka > kb:
            continue
        df_diff["diff__{}__from__{}".format(kb, ka)] = df[ka] - df[kb]
    return df_diff


def _diff(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    output_columns: Union[str, List[str]] = None,
):
    temp_df = pd.DataFrame(index=df.index)
    for c, oc in zip(columns, output_columns):
        temp_df[oc] = df[c].diff(periods=period)
    return temp_df


@func_group_by_key
def diff(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    id_columns: Union[str, List[str]] = None,
    output_columns: Union[str, List[str]] = None,
):
    columns = _ensure_list(columns)

    assert all([c in df.columns for c in columns])

    output_columns = (
        generate_output_columns("diff", period, columns=columns)
        if not output_columns
        else output_columns
    )

    df_diffed = _diff(df, columns=columns, period=period, output_columns=output_columns)

    return df_diffed


def _rolling(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    output_columns: Union[str, List[str]] = None,
    func=np.sum,
):
    temp_df = pd.DataFrame(index=df.index)
    for c, oc in zip(columns, output_columns):
        temp_df[oc] = df[c].rolling(window=period).apply(func=func, raw=True)
    return temp_df


@func_group_by_key
def rolling(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    period: int = 1,
    id_columns: Union[str, List[str]] = None,
    output_columns: Union[str, List[str]] = None,
    agg_func=np.sum,
):
    columns = _ensure_list(columns)

    assert all([c in df.columns for c in columns])

    output_columns = (
        generate_output_columns(
            "roll", period, str(agg_func.__qualname__), columns=columns
        )
        if not output_columns
        else output_columns
    )

    df_diffed = _rolling(
        df, columns=columns, period=period, output_columns=output_columns, func=agg_func
    )

    return df_diffed
