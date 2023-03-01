from typing import Dict

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn import pipeline as sk_pipe
from sklearn import preprocessing as sk_pp
from sklearn import decomposition as sk_decom
from sklearn import metrics as sk_me

from qbpo.kompete.godaddy.models import model_selection as qb_ms

def baseline_pipeline(
        df_tr:pd.DataFrame, df_va:pd.DataFrame, df_ts_feats:pd.DataFrame, df_en_feats:pd.DataFrame, 
        pipeline:sk_pipe.Pipeline, COL_TARGET:str, COL_EN:str, COL_TS:str, idx_to_info:Dict, recursive_delta:bool
    ):
    df_tr_ = df_tr.copy()
    df_va_ = df_va.copy()
    for df_ in [df_ts_feats, df_en_feats]:
        if df_ is not None:
            df_tr_ = pd.merge(df_tr_, df_, left_index=True, right_index=True)
            df_va_ = pd.merge(df_va_, df_, left_index=True, right_index=True)

    df_tr_.dropna(inplace=True, axis=1, how="all") # empty columns
    # df_tr_.dropna(inplace=True) # top rows
    
    VA_TS = qb_ms.get_future_timestamps(df_va)
    print (VA_TS)
    dfs = []
    for idx, ts in tqdm(enumerate(VA_TS)):
        features_en = list(df_en_feats.columns) if df_en_feats is not None else []
        features = idx_to_info[idx]["features"] + list(df_tr.columns) + features_en

        df_tr_X = df_tr_[[c for c in df_tr_ if c in features and c != COL_TARGET]]

        pipeline.fit(df_tr_X, df_tr_[COL_TARGET])
        
        selection = df_va_[COL_TS] == ts
        df_va_selected = df_va_[selection]
        df_va_X = df_va_selected[[c for c in df_tr_ if c in features and c != COL_TARGET]]
        
        df_ = df_va_selected[[COL_TARGET, COL_EN, COL_TS]]
        y_pred = pipeline.predict(df_va_X)
        
        df_["y_pred"] = y_pred

        if recursive_delta:
            if dfs:
                df_prev = dfs[-1] 
                df_["prev_y_pred"] = df_.apply(lambda x : df_prev.loc[df_prev[COL_EN] == x[COL_EN], "y_pred"].item(), axis=1)
                df_["y_pred"] = df_["prev_y_pred"] + df_["y_pred"]
                df_.drop("prev_y_pred", axis=1, inplace=True)
        
        dfs.append(df_)
    
    df_results = pd.concat(dfs)
    
    return df_results

def xgb_pipeline(df_tr, df_va, df_ts_feats, df_en_feats, idx_to_info, N_COMPONENTS_PCA_MBD_PER_CFIPS=12, TARGET_COLUMN="active"):

    df_tr_ = df_tr.copy()
    df_va_ = df_va.copy()
    for df_ in [df_ts_feats, df_en_feats]:
        if df_ is not None:
            df_tr_ = pd.merge(df_tr_, df_, left_index=True, right_index=True)
            df_va_ = pd.merge(df_va_, df_, left_index=True, right_index=True)

    df_tr_.dropna(inplace=True, axis=1, how="all") # empty columns

    # df_va_ = df_va_[[c for c in df_va_.columns if c not in ["microbusiness_density", "active"]]]

    ohe = sk_pp.OneHotEncoder(sparse=False)
    ohe.fit(df_tr["state"].to_frame())
   
    df_cfips = df_tr_.pivot(index="cfips", columns="first_day_of_month", values="microbusiness_density")

    pca_mbd_per_cfips = sk_decom.PCA(n_components=N_COMPONENTS_PCA_MBD_PER_CFIPS)
    cfips_pca = pca_mbd_per_cfips.fit_transform(df_cfips)
    df_cfips_pca = pd.DataFrame(cfips_pca, index=df_cfips.index, columns=["pca_cfips_{}".format(idx) for idx in range(0, N_COMPONENTS_PCA_MBD_PER_CFIPS)])

    df_tr_ = pd.merge(df_tr_, df_cfips_pca, left_on="cfips", right_index=True)
    df_va_ = pd.merge(df_va_, df_cfips_pca, left_on="cfips", right_index=True)

    for idx, info in tqdm(idx_to_info.items()):

        features = info["features"]

        pca = sk_decom.PCA (n_components=int(0.2 * len(features)), random_state=42)
        
        features_selected = [c for c in df_tr_.columns if c in features]
        selection = df_tr_[features_selected].dropna(inplace=False).index# with no edge rows (with NA)
        df_tr_special_nona = df_tr_.loc[selection]
        pca_feats = pca.fit_transform(df_tr_special_nona[features_selected].values)

        cfips_pca_feats = df_tr_special_nona[[c for c in df_tr_special_nona.columns if c.startswith("pca_cfips_")]].values

        ohe_feats = ohe.transform(df_tr_special_nona["state"].to_frame())
        
        all_feats = np.hstack((pca_feats, cfips_pca_feats, ohe_feats))

        # print (all_feats.shape, pca_feats.shape, cfips_pca_feats.shape, ohe_feats.shape)

        y = df_tr_special_nona[TARGET_COLUMN]

        print ("fitting model")

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=16, max_leaves=16, 
            learning_rate=0.3, 
            verbosity=1, n_jobs=8, random_state=42, 
            eval_metric=sk_me.mean_absolute_percentage_error, gpu_id=0)

        model.fit(all_feats, y)

        print ("model fit")

        idx_to_info[idx]["model"] = {
            "pca" : pca, 
            "model" : model,
            "features" : features_selected
        }

    df_result_list = []
    idx_to_fdom = {idx: fdom for idx, fdom in enumerate(sorted(df_va_.first_day_of_month.unique()))}
    for idx, info in tqdm(idx_to_info.items()):
        
        model_ = info["model"]

        features = model_["features"]
        pca = model_["pca"]
        model = model_["model"]

        fdom = idx_to_fdom[idx]

        features_selected = [c for c in df_tr_.columns if c in features]
        
        df_va_special_nona = df_va_[df_va_.first_day_of_month == fdom]    

        pca_feats_va = pca.transform(df_va_special_nona[features])

        cfips_pca_feats_va = df_va_special_nona[[c for c in df_va_special_nona.columns if c.startswith("pca_cfips_")]].values

        ohe_feats_va = ohe.transform(df_va_special_nona["state"].to_frame())

        all_feats_va = np.hstack((pca_feats_va, cfips_pca_feats_va, ohe_feats_va))
        
        # print (all_feats_va.shape, pca_feats_va.shape, cfips_pca_feats_va.shape, ohe_feats_va.shape)

        pred_y = model.predict(all_feats_va)

        df_result = df_va_special_nona[["active", "cfips", "first_day_of_month"]]
        df_result.loc[:, "y_pred_xgb"] = pred_y
        
        df_result_list.append(df_result)

    df_results = pd.concat(df_result_list)

    return df_results, ohe, pca_mbd_per_cfips