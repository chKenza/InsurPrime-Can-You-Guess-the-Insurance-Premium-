# feature_eng.py

import numpy as np
import pandas as pd

# This file is created so that the function add_engineered_features() can be imported easly
# usage : from feature_eng import add_engineered_features

def add_engineered_features(X_df, is_train=True, feature_meta=None,
                            rare_threshold=200):
    """   
    - Aggregate SURFACE*, KAPITAL*, NBBAT*
    - Prevention, weather, risk, and grouping features.
    - Derived features (log exposure, complexity).
    - Uses feature_meta only for rare-category grouping so train/test mappings match.
    """
    if feature_meta is None:
        feature_meta = {}

    X = X_df.copy()

    # exposure
    if "ANNEE_ASSURANCE" in X.columns:
        # log exposure
        X["LOG_EXPOSURE"] = np.log1p(X["ANNEE_ASSURANCE"].astype(float))


    # SURFACE aggregates
    surface_cols = [c for c in X.columns if c.startswith("SURFACE")]
    if surface_cols:
        # Cast to numeric for arithmetic; categories/strings -> numbers
        surface_num = X[surface_cols].apply(pd.to_numeric, errors="coerce")

        X["TOT_SURFACE"] = surface_num.fillna(0).sum(axis=1)
        X["SURFACE_NONZERO_COUNT"] = (surface_num.fillna(0) > 0).sum(axis=1)
        X["SURFACE_MAX"] = surface_num.fillna(0).max(axis=1)

    # NBBAT aggregates
    nbbat_cols = [c for c in X.columns if c.startswith("NBBAT")]
    if nbbat_cols:
        nbbat_num = X[nbbat_cols].apply(pd.to_numeric, errors="coerce")

        X["TOT_BUILDINGS"] = nbbat_num.fillna(0).sum(axis=1)
        if "TOT_SURFACE" in X.columns:
            X["AVG_SURFACE_PER_BUILDING"] = (
                X["TOT_SURFACE"] / (1.0 + X["TOT_BUILDINGS"])
            )


    # Simple complexity proxy: more buildings + more surface types = more complex farm
    if ("TOT_BUILDINGS" in X.columns) and ("SURFACE_NONZERO_COUNT" in X.columns):
        X["COMPLEXITY_INDEX"] = X["TOT_BUILDINGS"] + X["SURFACE_NONZERO_COUNT"]

    # KAPITAL aggregates
    kap_cols = [c for c in X.columns if c.startswith("KAPITAL")]
    if kap_cols:
        kap_num = X[kap_cols].apply(pd.to_numeric, errors="coerce")
        X["TOT_KAPITAL"] = kap_num.fillna(0).sum(axis=1)
        X["KAPITAL_NONZERO_COUNT"] = (kap_num.fillna(0) > 0).sum(axis=1)
        X["LOG_TOT_KAPITAL"] = np.log1p(X["TOT_KAPITAL"])


    # Prevention features (PREV*)
    prev_cols = [c for c in X.columns if c.upper().startswith("PREV")]
    if prev_cols:
        X["N_PREVENTION"] = X[prev_cols].notna().sum(axis=1)
        X["NO_PREVENTION_FLAG"] = (X["N_PREVENTION"] == 0).astype(int)

    # Weather features
    # DRY_INDEX = dry days / (1 + rain)
    if {"NBJDRY_MM_A", "RR_MM_A"}.issubset(X.columns):
        X["DRY_INDEX"] = X["NBJDRY_MM_A"] / (1.0 + X["RR_MM_A"].clip(lower=0))

    # Wind risk: high wind zone (ZONE_VENT == 3)
    if "ZONE_VENT" in X.columns:
        zone_vent_vals = pd.to_numeric(X["ZONE_VENT"], errors="coerce")
        X["VENT_HIGH"] = (zone_vent_vals == 3).astype(int)

    # Activity / Vocation grouping (rare categories)
    # ACTIVIT2
    if "ACTIVIT2" in X.columns:
        # Work as string to avoid categorical headaches
        activ = X["ACTIVIT2"].astype("string")

        if is_train:
            vc_act = activ.value_counts()
            rare_act = vc_act[vc_act < rare_threshold].index.astype(str)
            feature_meta["activit2_rare"] = list(rare_act)
        else:
            rare_act = feature_meta.get("activit2_rare", [])

        X["ACTIVIT2_GRP"] = activ.where(~activ.isin(rare_act), "OTHER").astype("category")

    # VOCATION
    if "VOCATION" in X.columns:
        voc = X["VOCATION"].astype("string")

        if is_train:
            vc_voc = voc.value_counts()
            rare_voc = vc_voc[vc_voc < rare_threshold].index.astype(str)
            feature_meta["vocation_rare"] = list(rare_voc)
        else:
            rare_voc = feature_meta.get("vocation_rare", [])

        X["VOCATION_GRP"] = voc.where(~voc.isin(rare_voc), "OTHER").astype("category")

        
    # Contract age / "new business"
    if "ANCIENNETE" in X.columns:
        X["IS_NEW_CONTRACT"] = (X["ANCIENNETE"].astype(float) < 1).astype(int)

    # Risk-related features (RISK*)
    risk_cols = [c for c in X.columns if c.startswith("RISK")]
    if risk_cols:
        X["RISK_ALERTS"] = X[risk_cols].notna().sum(axis=1)


    # Simple interactions

    # Surface x Prevention
    if ("TOT_SURFACE" in X.columns) and ("N_PREVENTION" in X.columns):
        X["SURFACE_PREVENTION"] = X["TOT_SURFACE"] * X["N_PREVENTION"]

    # Capital x Zone
    if ("LOG_TOT_KAPITAL" in X.columns) and ("ZONE" in X.columns):
        zone_codes = pd.to_numeric(X["ZONE"], errors="coerce")
        X["KAPITAL_ZONE"] = X["LOG_TOT_KAPITAL"] * (zone_codes.fillna(0))

    # Buildings x Dryness
    if ("TOT_BUILDINGS" in X.columns) and ("DRY_INDEX" in X.columns):
        X["BUILDINGS_DRY"] = X["TOT_BUILDINGS"] * X["DRY_INDEX"]

    return X, feature_meta