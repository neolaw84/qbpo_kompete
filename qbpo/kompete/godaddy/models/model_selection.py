from typing import List

from tqdm.auto import tqdm

import pandas as pd

def get_future_timestamps(df_test:pd.DataFrame):
    TS_TEST_ALL = df_test.first_day_of_month.unique()
    TS_TEST_ALL.sort()
    return TS_TEST_ALL

def get_available_features(df_all:pd.DataFrame, df_ts_feats:pd.DataFrame, ts_test_start:str="2022-11-01"):
    TS_ALL = df_all.first_day_of_month.unique()
    TS_TEST_ALL = [ts for ts in TS_ALL if ts >= ts_test_start]

    idx_to_info = {}
    
    for idx, fdom in tqdm(enumerate(TS_TEST_ALL)):
        selection = df_all.first_day_of_month == fdom
        selection_idx = df_all.index[selection]
        _df = df_ts_feats.loc[selection_idx]
        columns_avail = [c for c in _df.columns if not _df[c].isnull().any()]
        idx_to_info[idx] = {
            "fdom": fdom,
            "features": columns_avail,
        }

    return idx_to_info