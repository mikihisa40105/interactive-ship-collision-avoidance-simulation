"""Helpers for comparing gamma and log-normal passage-position models."""

import pandas as pd

from .losses import GammaMetrics1D, LogNormalMetrics1D


def _evaluate_one(model, X, y, metrics_obj, distribution_name, side_name, batch_size=256, **metrics_kwargs):
    pred = model.predict(X, batch_size=batch_size)
    k_params = model.count_params()
    result = metrics_obj.from_predictions(y, pred, k_params=k_params, **metrics_kwargs)
    result.update(
        {
            "side": side_name,
            "distribution": distribution_name,
            "k_params": k_params,
        }
    )
    return result


def compare_models_stb_prt(
    model_gamma_stb,
    data_gamma_stb,
    model_lognorm_stb,
    data_lognorm_stb,
    model_gamma_prt,
    data_gamma_prt,
    model_lognorm_prt,
    data_lognorm_prt,
    batch_size=256,
):
    """Compare STB/PRT gamma and log-normal models using NLL, AIC, and BIC.

    The data tuples follow the historical data convention where X is index 4 and y is
    index 5.
    """
    rows = [
        _evaluate_one(
            model_gamma_stb,
            data_gamma_stb[4],
            data_gamma_stb[5],
            GammaMetrics1D(),
            "Gamma",
            "STB",
            batch_size=batch_size,
        ),
        _evaluate_one(
            model_lognorm_stb,
            data_lognorm_stb[4],
            data_lognorm_stb[5],
            LogNormalMetrics1D(),
            "LogNormal",
            "STB",
            batch_size=batch_size,
        ),
        _evaluate_one(
            model_gamma_prt,
            data_gamma_prt[4],
            data_gamma_prt[5],
            GammaMetrics1D(),
            "Gamma",
            "PRT",
            batch_size=batch_size,
        ),
        _evaluate_one(
            model_lognorm_prt,
            data_lognorm_prt[4],
            data_lognorm_prt[5],
            LogNormalMetrics1D(),
            "LogNormal",
            "PRT",
            batch_size=batch_size,
        ),
    ]

    df = pd.DataFrame(rows)
    if {"AIC", "BIC", "N"}.issubset(df.columns):
        df["AIC_per_sample"] = df["AIC"] / df["N"]
        df["BIC_per_sample"] = df["BIC"] / df["N"]

    preferred_cols = [
        "side",
        "distribution",
        "N",
        "k_params",
        "avg_logL",
        "mean_NLL",
        "total_logL",
        "AIC",
        "BIC",
        "AIC_per_sample",
        "BIC_per_sample",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    df_sorted = df.sort_values(["side", "mean_NLL"], ascending=[True, True]).reset_index(drop=True)
    df_sorted["is_best_in_side"] = False
    for _, group in df_sorted.groupby("side"):
        best_idx = group["mean_NLL"].idxmin()
        df_sorted.loc[best_idx, "is_best_in_side"] = True
    return df_sorted
