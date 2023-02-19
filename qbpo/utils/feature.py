import functools
import itertools
import inspect
from typing import Union, List

from tqdm.auto import tqdm

import pandas as pd

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

def func_group_by_key(func=None, id_column_arg_name:str="id_columns"):
    @functools.wraps(func)
    def wrapper_func_group_by_key(*args, **kwargs):
        id_columns = get_arg(func, id_column_arg_name, *args, **kwargs)
        df = get_arg(func, "df", *args, **kwargs)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        if id_columns:
            dfs = df.groupby(by=id_columns)
            _dfs = []
            for _, _df in tqdm(dfs):
                args = (a for a in args if a is not df)
                kwargs = {k: v for k, v in kwargs.items() if v is not df}
                temp_df = func(
                    _df, *args, **kwargs
                )
                _dfs.append(temp_df)
            df_feature = pd.concat(_dfs)
        else:
            df_feature = func(
                *args, **kwargs
            )

        return df_feature
    return wrapper_func_group_by_key

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
    if isinstance(columns, str):
        columns = [columns]

    for c in columns:
        assert c in df.columns

    if not output_columns:
        output_columns = ["{c}__shift__{p}".format(c=c, p=period) for c in columns]

    df_shifted = _shift(
        df, columns=columns, period=period, output_columns=output_columns
    )

    return df_shifted


def diff(df: pd.DataFrame, columns: Union[str, List[str]], mirror:bool=False):
    df_diff = pd.DataFrame(index=df.index)
    pairs = itertools.product(columns, columns)
    for ka, kb in tqdm(pairs):
        if ka == kb: continue
        if mirror and ka > kb: continue
        df_diff["diff__{}__from__{}".format(kb, ka)] = df[ka] - df[kb]
    return df_diff