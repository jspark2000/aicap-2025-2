from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_BASE_SENTINELS: set[float] = {
    7,
    8,
    9,
    77,
    88,
    99,
    777,
    888,
    999,
    7777,
    8888,
    9999,
}

DEFAULT_SENTINELS: set[float] = set(_BASE_SENTINELS)
DEFAULT_SENTINELS.update(float(v) for v in _BASE_SENTINELS)


def load_khanes_data(
    path: str | bytes,
    usecols: Sequence[str] | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    KNHANES CSV 파일을 읽고 이름 없는 인덱스 컬럼을 제거합니다.
    """
    df = pd.read_csv(path, usecols=usecols, low_memory=low_memory)
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def replace_numeric_sentinels(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    sentinel_values: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    KNHANES 상징값(예: 8, 88, 8888)을 NaN으로 변환합니다.
    """
    sentinels = set(sentinel_values or DEFAULT_SENTINELS)
    target_cols = (
        list(columns)
        if columns is not None
        else df.select_dtypes(include=["number"]).columns.tolist()
    )
    for col in target_cols:
        df[col] = df[col].replace(list(sentinels), np.nan)
    return df


def cap_iqr_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    whisker_width: float = 1.5,
) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 IQR 기반 whisker 범위를 벗어나는 이상치를 제한(capping)합니다.
    """
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - whisker_width * iqr
        upper = q3 + whisker_width * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def interpolate_numeric_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    order_col: str,
) -> pd.DataFrame:
    """
    order_col 기준으로 정렬한 후 수치형 변수들에 선형 보간법을 적용합니다.
    """
    sorted_idx = df[order_col].sort_values().index
    interpolated = (
        df.loc[sorted_idx, columns].interpolate(method="linear", limit_direction="both")
    )
    df.loc[sorted_idx, columns] = interpolated
    return df


def standardize_features(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    지정된 변수 컬럼들에 대해 Z-score 표준화를 수행합니다.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in columns:
        series = df[col]
        mean = series.mean()
        std = series.std(ddof=0)
        if pd.isna(mean) or pd.isna(std) or std == 0:
            df[col] = 0.0
            stats[col] = {"mean": mean, "std": std}
            continue
        df[col] = (series - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return df, stats


def compute_metabolic_flag(
    df: pd.DataFrame,
    diag_cols: Sequence[str] = ("DI1_dg", "DE1_dg", "DI2_dg"),
) -> pd.Series:
    """
    진단 컬럼들 중 하나라도 의사 진단 질환이 있으면 1을 반환합니다.
    """
    missing = [col for col in diag_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing diagnosis columns: {missing}")
    diag_df = df[list(diag_cols)].fillna(0)
    return (diag_df == 1).any(axis=1).astype(int)


def filter_non_smoking_non_drinking(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    smoker_col: str = "sm_presnt",
    drinking_freq_col: str = "dr_month",
    lifetime_drink_col: str = "BD1",
    min_age: int = 20,
    max_age: int = 64,
) -> pd.DataFrame:
    """
    비흡연·비음주 성인 중 20-64세 대상자에 대한 연구 포함 기준을 적용합니다.
    """
    mask = (
        df[age_col].between(min_age, max_age)
        & (df[smoker_col] == 0)
        & (df[drinking_freq_col] == 0)
        & (df[lifetime_drink_col] == 2)
    )
    return df.loc[mask].copy()
