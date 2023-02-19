import pathlib
import warnings

from glob import glob
from datetime import datetime as dt

from typing import Union, Sequence, List, Tuple

from tqdm.auto import tqdm

import opendatasets as od

import pandas as pd

tqdm.pandas()

THIS_FILE_PATH = pathlib.Path(__file__)
PROJECT_ROOT = THIS_FILE_PATH.parent

DEFAULT_KAGGLE_KEY_PATH = "./kaggle.json"


def download_kaggle_dataset(
    dataset_id_or_url,
    data_dir: str = ".",
    force: bool = False,
    dry_run: bool = False,
    kaggle_key_json_path: str = DEFAULT_KAGGLE_KEY_PATH,
):
    """_Download kaggle dataset from the given dataset id or url_

    Args:
        dataset_id_or_url (_type_): _kaggle dataset id or https url to kaggle dataset_
        data_dir (str, optional): _data directory for the downloaded data to go to_. Defaults to ".".
        force (bool, optional): _whether to force download if it exists_. Defaults to False.
        dry_run (bool, optional): _whether to dry-run the download_. Defaults to False.
        kaggle_key_json_path (str, optional): _path to `kaggle.json` key file_. Defaults to DEFAULT_KAGGLE_KEY_PATH.
    """
    data_path = pathlib.Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    kaggle_key_path = pathlib.Path(kaggle_key_json_path)

    assert kaggle_key_path.exists()

    _effective_kaggle_key_path = pathlib.Path(DEFAULT_KAGGLE_KEY_PATH)

    # if effective one exists and not the same file, back it up
    if _effective_kaggle_key_path.exists() and not _effective_kaggle_key_path.samefile(
        kaggle_key_path
    ):
        warnings.warn(
            "Renaming file at {} to {}".format(
                _effective_kaggle_key_path,
                _effective_kaggle_key_path.with_suffix(".json.bak"),
            )
        )
        new_file_name = "kaggle_{}".format(dt.now().strftime("%y-%m-%d-%H-%M-%S"))
        _effective_kaggle_key_path.rename(
            _effective_kaggle_key_path.with_name(new_file_name)
        )
        _effective_kaggle_key_path = pathlib.Path(DEFAULT_KAGGLE_KEY_PATH)

    # now, if effective one still exists, it means it is the same file
    # if it does not, we need to symlink it
    if not _effective_kaggle_key_path.exists():
        _effective_kaggle_key_path.symlink_to(kaggle_key_path)

    od.download(dataset_id_or_url, data_dir=data_path, force=force, dry_run=dry_run)


def read_glob_to_df(
    glob_exp: str,
    file_type: str = "csv",
    sep: str = None,
    delimiter: str = None,
    header: Union[int, Sequence[int]] = None,
    names: List[str] = None,
    index_col: Union[int, str, Sequence[Union[str, int]]] = None,
    usecols: Union[List[str], Tuple[str], Sequence[int]] = None,
    dtype=None,
    **kwargs
) -> pd.DataFrame:
    files = glob(glob_exp)
    dfs = []
    file_type_to_reader = {"csv": pd.read_csv}
    assert file_type in file_type_to_reader.keys()
    reader_func = file_type_to_reader.get(file_type)
    for f in tqdm(files):
        _df = reader_func(
            f,
            sep=sep,
            delimiter=delimiter,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            **kwargs
        )
        dfs.append(_df)
    df = pd.concat(dfs)
    return df


DEFAULT_FORMAT_DATETIME = "%Y-%m-%d"


def _impute_ts(df: pd.DataFrame, min_ts, max_ts, ts_column, delta="MS"):
    all_dates = pd.date_range(start=min_ts, end=max_ts, freq=delta).to_series()
    all_dates.name = ts_column
    temp_df = df.merge(
        all_dates, how="right", left_on=ts_column, right_on=ts_column, sort=True
    )
    return temp_df


def add_ts_column(
    df: pd.DataFrame,
    ts_column,
    new_ts_column: str = "",
    ts_format: str = DEFAULT_FORMAT_DATETIME,
    inplace: bool = False,
):
    """_create new timestamp column from given timestamps in `ts_column` and add it into the dataframe_

    Args:
        df (pd.DataFrame): _description_
        ts_column (_type_): _description_
        new_ts_column (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _None if inplace; else the dataframe_
    """

    ds = pd.to_datetime(df.first_day_of_month, format=ts_format)
    if not new_ts_column:
        new_ts_column = ts_column + "_ts"

    temp_df = df if inplace else df.copy()

    temp_df[new_ts_column] = ds

    return temp_df


def fill_ts(
    df: pd.DataFrame, id_columns: Union[str, List[str]]=None, ts_column: str = "", delta="MS"
):
    """_ensure dataframe includes all timestamps_

    Args:
        df (pd.DataFrame): _source dataframe_
        id_columns (Union[str, List[str]], optional): _key columns of the dataframe_. Defaults to None.
        ts_column (str, optional): _timestamp column_. Defaults to "".
        delta (str, optional): _delta of timestamps to fill_. Defaults to "MS".

    Returns:
        _type_: _description_
    """
    min_ts = df.first_day_of_month_ts.min()
    max_ts = df.first_day_of_month_ts.max()

    if id_columns:
        dfs = df.groupby(by=id_columns)
        _dfs = []
        for _, _df in tqdm(dfs):
            temp_df = _impute_ts(
                _df, min_ts=min_ts, max_ts=max_ts, ts_column=ts_column, delta=delta
            )
            _dfs.append(temp_df)
        df_ts_impute = pd.concat(_dfs)
        df_ts_impute.reset_index(inplace=True, drop=True)
    else:
        df_ts_impute = _impute_ts(df, min_ts, max_ts, ts_column=ts_column, delta=delta)

    return df_ts_impute
