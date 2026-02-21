# data_preprocessing.py

import numpy as np
import pandas as pd

# This file is created so that the function preprocess_data() can be imported easly
# usage : from data_preprocessing import preprocess_data

def preprocess_data(X_df, y_df=None, is_train=True, fitted_transformers=None):
    """
    Data cleaning:
      - Drop ID, constant cols, ultra-sparse cols
      - Replace sentinel values for SURFACE/KAPITAL/RISK
      - Convert binary numerics to categorical
      - Convert coded numeric families (SURFACE, KAPITAL, RISK, ZONE) to categorical
      - Leave NaNs as NaNs (no imputation)
    """
    if fitted_transformers is None:
        fitted_transformers = {}

    X = X_df.copy()
    y = y_df.copy() if y_df is not None else None

    if is_train:
        # Drop ID
        if "ID" in X.columns:
            X = X.drop(columns=["ID"])
            fitted_transformers["drop_id"] = True

        # Drop single-value columns
        single_val_cols = X.columns[X.nunique(dropna=True) <= 1].tolist()
        X = X.drop(columns=single_val_cols)
        fitted_transformers["single_val_cols"] = single_val_cols

        # Drop ultra-sparse columns
        na_rate = X.isna().mean()
        ultra_sparse_cols = na_rate[na_rate > 0.995].index.tolist()
        X = X.drop(columns=ultra_sparse_cols)
        fitted_transformers["ultra_sparse_cols"] = ultra_sparse_cols

        # Drop known multicollinear columns
        fixed_drop_cols = [
            "NBBAT4", "NBBAT13", "NBBAT10",
            "EQUIPEMENT7"
        ]
        fixed_drop_cols = [c for c in fixed_drop_cols if c in X.columns]
        if fixed_drop_cols:
            X = X.drop(columns=fixed_drop_cols)
        fitted_transformers["fixed_drop_cols"] = fixed_drop_cols

        # Convert fake codes to NaN, e.g., sentinel values in SURFACE/KAPITAL/RISK
        sentinel_cols = [c for c in X.columns if c.startswith(("SURFACE", "KAPITAL", "RISK"))]
        X[sentinel_cols] = X[sentinel_cols].replace({1000: np.nan, -1: np.nan, -3: np.nan})
        fitted_transformers["sentinel_cols"] = sentinel_cols

        # Convert binary numeric columns to categorical
        num_cols = X.select_dtypes(include=[np.number]).columns
        binary_num_cols = [c for c in num_cols if X[c].nunique(dropna=True) == 2]
        X[binary_num_cols] = X[binary_num_cols].astype("category")
        fitted_transformers["binary_num_cols"] = binary_num_cols

        # Convert Coded Numerics to Categorical
        # coded_prefixes = ("SURFACE", "KAPITAL", "RISK")
        coded_prefixes = ("RISK")
        coded_cols = [c for c in X.columns if c.startswith(coded_prefixes)]
        if "ZONE" in X.columns:
            coded_cols.append("ZONE")
        coded_cols = list(dict.fromkeys(coded_cols))

        X[coded_cols] = X[coded_cols].astype("category")
        fitted_transformers["coded_cols"] = coded_cols

        # Convert obj to category (memory effcient)
        obj_cols = X.select_dtypes(include=["object"]).columns
        X[obj_cols] = X[obj_cols].astype("category")
        fitted_transformers["obj_cols_to_category"] = list(obj_cols)

        if y is not None:
            if "CM" in y.columns:
                y["CM"] = y["CM"].clip(lower=0)
            if "CHARGE" in y.columns:
                y["CHARGE"] = y["CHARGE"].clip(lower=0)

        return X, y, fitted_transformers

    else:
        # we apply the train time decisions

        # Drop ID
        if fitted_transformers.get("drop_id", False) and "ID" in X.columns:
            X = X.drop(columns=["ID"])

        # Drop same constant & ultra-sparse columns
        for key in ["single_val_cols", "ultra_sparse_cols"]:
            drop_cols = fitted_transformers.get(key, [])
            existing = [c for c in drop_cols if c in X.columns]
            if existing:
                X = X.drop(columns=existing)

        # Drop the same multicollinear columns as in training
        fixed_drop_cols = fitted_transformers.get("fixed_drop_cols", [])
        existing_fixed = [c for c in fixed_drop_cols if c in X.columns]
        if existing_fixed:
            X = X.drop(columns=existing_fixed)

        # Replace sentinel values in same sentinel_cols
        sentinel_cols = fitted_transformers.get("sentinel_cols", [])
        existing_sentinel = [c for c in sentinel_cols if c in X.columns]
        if existing_sentinel:
            X[existing_sentinel] = X[existing_sentinel].replace({1000: np.nan, -1: np.nan, -3: np.nan})

        # Binary numeric to category
        binary_num_cols = fitted_transformers.get("binary_num_cols", [])
        existing_binary = [c for c in binary_num_cols if c in X.columns]
        if existing_binary:
            X[existing_binary] = X[existing_binary].astype("category")

        # Coded families tocategory
        coded_cols = fitted_transformers.get("coded_cols", [])
        existing_coded = [c for c in coded_cols if c in X.columns]
        if existing_coded:
            X[existing_coded] = X[existing_coded].astype("category")

        # Object to category
        obj_cols = X.select_dtypes(include=["object"]).columns
        X[obj_cols] = X[obj_cols].astype("category")

        return X, y, fitted_transformers