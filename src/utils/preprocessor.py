from itertools import product
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandas.testing as pdt
from omegaconf import DictConfig
from sklearn import preprocessing
from tabulate import tabulate

from src.utils import tabulate_wide


def preprocess(data: Union[np.ndarray, pd.DataFrame], cfg: DictConfig) -> Union[np.ndarray, pd.DataFrame]:
    """
    Preprocess the input data according to the configuration.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.
    cfg : DictConfig
        Configuration for preprocessing.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Preprocessed data.
    """
    io_cfg = cfg.data
    if io_cfg.moving_average > 1:
        data = moving_average(data, window=io_cfg.moving_average)
    if io_cfg.logarithm:
        data = log(data)
    if io_cfg.scaling_kind == 'zscore':
        data = scale(data)
    elif io_cfg.scaling_kind == 'minmax':
        data = minmax_scale(data)
    elif io_cfg.scaling_kind == 'normalize':
        data = normalize(data)
    if io_cfg.whitenoise > 0:
        seed = getattr(cfg, 'seed', None)
        data = add_whitenoise(data, std=io_cfg.whitenoise, seed=seed)

    return data


def moving_average(X: Union[np.ndarray, pd.DataFrame], window: int) -> Union[np.ndarray, pd.DataFrame]:
    """
    Compute trailing moving average along the first axis (time axis).

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.
    window : int
        Window length (> 0). A trailing window is used.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Moving-averaged data with the same type as input.
        Shape is (n - window + 1, d1, d2, ..., dM).
    """
    n = X.shape[0]
    if not isinstance(window, int) or window <= 0 or window > n:
        raise ValueError('`window` must be a positive integer less than or equal to the number of samples (n).')

    # pandas path DataFrame
    if isinstance(X, pd.DataFrame):
        if window == 1:
            return X.copy()
        # min_periods=window to align with the specification (leading NaNs)
        rolled = X.rolling(window=window, min_periods=window).mean()
        return rolled.iloc[window - 1 :, :]

    # numpy path
    if not isinstance(X, np.ndarray):
        raise TypeError('`X` must be a np.ndarray or a pd.DataFrame.')

    if X.ndim == 0:
        raise ValueError('`X` must have at least one dimension (time axis at axis=0).')

    # Promote dtype to float for proper averaging and NaN fill
    x_float = X.astype(np.result_type(X.dtype, np.float64), copy=False)

    # Cumulative sum along axis=0
    csum = np.cumsum(x_float, axis=0)

    # sum over each trailing window [t-window+1, t]
    # For t = window-1, subtract "zero-sum" (prepend zeros)
    # Build shifted cumulative sum with a leading zero-slice
    zero_pad = np.zeros_like(csum[:1, ...])
    csum_shifted = np.concatenate([zero_pad, csum[:-window, ...]], axis=0)  # shape (n-window+1, ...)
    window_sums = csum[window - 1 :, ...] - csum_shifted  # shape (n-window+1, ...)
    return window_sums / window


def log(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Apply a logarithmic transform elementwise.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Log-transformed data with the same shape/type as input.

    Rules
    -----
    1) If min(X) > 0: y = log(X)
    2) If min(X) >= 0: y = log1p(X) # for zeros / count-like data
    3) If any X < 0: raise ValueError (do not auto-shift to avoid distortion)

    Notes
    -----
    - NaN values are propagated as-is.
    - The output is promoted to floating-point type.
    - For pandas objects, index/columns are preserved.
    """

    # Switch function
    def _apply_log(arr: np.ndarray) -> np.ndarray:
        if np.isnan(arr).all():
            raise ValueError('All values are NaN; log transform is undefined.')
        arr = arr.astype(np.result_type(arr.dtype, np.float64), copy=False)
        amin = np.nanmin(arr)
        if amin > 0:
            return np.asarray(np.log(arr), dtype=np.float64)
        if amin >= 0:
            return np.asarray(np.log1p(arr), dtype=np.float64)
        raise ValueError(
            'Negative values are present in the data for log transformation. '
            'Consider preprocessing the data according to its characteristics,'
            'e.g., X - X.min() + ε, or sign(x) * log(1 + |x|), etc.'
        )

    if isinstance(X, pd.DataFrame):
        out = _apply_log(X.to_numpy())
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    elif isinstance(X, np.ndarray):
        return _apply_log(X)
    else:
        raise TypeError('`X` must be one of numpy.ndarray or pandas DataFrame.')


def scale(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Compute z-score normalization along the first axis (time axis).

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Scaled data with the same shape/type as input.

    Notes
    -----
    - If missing values (NaN) are present, a ValueError is raised.
    - The dtype is promoted to at least float64 for calculations.
    """
    if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
        raise TypeError('`X` must be one of numpy.ndarray or pandas DataFrame.')

    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(copy=True)
    else:
        arr = np.array(X, copy=True)

    # Promote dtype
    arr = arr.astype(np.result_type(arr.dtype, np.float64), copy=True)

    # Check for NaNs
    if np.isnan(arr).any():
        raise ValueError('NaN values detected. Please impute or drop NaNs before calling scale().')

    n = arr.shape[0]
    rest = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    arr2d = arr.reshape(n, rest)
    out2d = preprocessing.scale(arr2d)
    out = out2d.reshape(arr.shape)

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out


def minmax_scale(
    X: Union[np.ndarray, pd.DataFrame],
    feature_range: Tuple = (-1, 1),
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Compute min-max scaling along the first axis (time axis).

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.
    feature_range : Tuple
        Desired range of transformed data. Default is (0, 1).

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Scaled data with the same shape/type as input.
    """
    # Type check
    if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
        raise TypeError('`X` must be one of numpy.ndarray or pandas DataFrame.')

    # feature_range validation
    if (not isinstance(feature_range, tuple)) or len(feature_range) != 2:
        raise ValueError('`feature_range` must be a tuple of length 2, e.g., (0, 1).')

    fr_min, fr_max = feature_range
    try:
        fr_min = float(fr_min)
        fr_max = float(fr_max)
    except (TypeError, ValueError):
        raise ValueError('`feature_range` entries must be numeric.')
    if fr_min >= fr_max:
        raise ValueError('`feature_range[0]` must be strictly less than `feature_range[1]`.')

    # Extract array (copy to avoid mutating caller's data) and promote dtype
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(copy=True)
    else:
        arr = np.array(X, copy=True)

    arr = arr.astype(np.result_type(arr.dtype, np.float64), copy=False)

    # NaN check (policy aligned with z-score `scale`)
    if np.isnan(arr).any():
        raise ValueError('NaN values detected. Please impute or drop NaNs before calling minmax_scale().')

    # Delegate to sklearn (per-feature scaling along axis=0)
    n = arr.shape[0]
    rest = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    arr2d = arr.reshape(n, rest)
    out2d = preprocessing.minmax_scale(arr2d, feature_range=(fr_min, fr_max))
    out = out2d.reshape(arr.shape)

    # Re-wrap as DataFrame if needed
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out


def normalize(X: Union[np.ndarray, pd.DataFrame], norm: str = 'l2') -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalize samples to unit norm along the first axis (time axis).

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.
    norm : str
        Norm to use. One of {'l1', 'l2', 'max'}.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Normalized data with the same shape/type as input.

    Notes
    -----
    - If a sample has zero norm, it is left unchanged (all-zero).
    - If missing values (NaN) are present, a ValueError is raised.
    - The dtype is promoted to at least float64 for calculations.
    """
    if norm not in {'l1', 'l2', 'max'}:
        raise ValueError("`norm` must be one of {'l1', 'l2', 'max'}.")

    # pandas DataFrame path
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(copy=True)
        arr = arr.astype(np.result_type(arr.dtype, np.float64), copy=False)

        if np.isnan(arr).any():
            raise ValueError('NaN values detected. Please impute or drop NaNs before calling normalize().')

        out = preprocessing.normalize(arr, norm=norm, axis=1, copy=True)
        return pd.DataFrame(out, index=X.index, columns=X.columns)

    # numpy ndarray path
    if isinstance(X, np.ndarray):
        if X.ndim == 0:
            raise ValueError('`X` must have at least one dimension (time axis at axis=0).')

        arr = np.array(X, copy=True)
        arr = arr.astype(np.result_type(arr.dtype, np.float64), copy=False)

        if np.isnan(arr).any():
            raise ValueError('NaN values detected. Please impute or drop NaNs before calling normalize().')

        n = arr.shape[0]
        rest = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        arr2d = arr.reshape(n, rest)
        out2d = preprocessing.normalize(arr2d, norm=norm, axis=1, copy=True)
        out = out2d.reshape(arr.shape)
        return out

    raise TypeError('`X` must be one of numpy.ndarray or pandas DataFrame.')


def add_whitenoise(
        X: Union[np.ndarray, pd.DataFrame],
        std: float,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Add Gaussian white noise to the data.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        M+1-th order tensor to preprocess.
        Shape is (T, d1, d2, ..., dM) if np.ndarray.
        For pd.DataFrame, index/columns are preserved.
    std : float
        Standard deviation of the Gaussian noise to add. Must be non-negative.
    seed : optional int
        Random seed for reproducibility.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Noised data with the same shape/type as input (values are float).
    """
    # std validation
    if std < 0.0:
        raise ValueError('`std` must be a non-negative float.')

    rng = np.random.default_rng(seed)
    def _add_noise(arr: np.ndarray, sigma: float) -> np.ndarray:
        # Promote dtype to float64 (or compatible) and copy
        out = np.array(arr, copy=True)
        out = out.astype(np.result_type(out.dtype, np.float64), copy=False)

        if out.size == 0 or sigma == 0.0:
            return out

        noise = rng.standard_normal(size=out.shape) * sigma
        # add only to finite entries; preserve NaN/inf as-is
        mask = np.isfinite(out)
        out[mask] += noise[mask]
        return out

    # pandas path
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(copy=True)
        out_arr = _add_noise(arr, std)
        return pd.DataFrame(out_arr, index=X.index, columns=X.columns)

    # numpy path
    if isinstance(X, np.ndarray):
        return _add_noise(X, std)

    raise TypeError('`X` must be one of numpy.ndarray or pandas DataFrame.')


def df2tts(
    df: pd.DataFrame,
    time_key: str,
    facets: List[str],
    values: str,
    sampling_rate: Optional[str] = 'D',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.MultiIndex]:
    """
    Convert a long-form DataFrame to tensor time series

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a timestamp column, facet columns, and a value column
    time_key : str
        Column name of timestamps
    facets : list
        List of column names to make tensor timeseries
    values : str
        Column name of target values
    sampling_rate : str
        Frequency for resampling, e.g., "7D", "12H", "H"
    start_date : str
        Start date for filtering (optional)
    end_date : str
        End date for filtering (optional)

    Returns
    -------
    tts : np.ndarray
        Tensor time series of shape (T, d1, d2, ..., dM), where T is the length.
    time_index : pd.DatetimeIndex
        Timestamps after rounding/resampling and date filtering.
    columns : pd.MultiIndex
        MultiIndex of shape (d1 * d2 * ... * dM,) for facets.
    """
    df = df.copy()
    df[time_key] = pd.to_datetime(df[time_key])
    if start_date is not None:
        df = df[lambda x: x[time_key] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[lambda x: x[time_key] <= pd.to_datetime(end_date)]

    # Resampling (if needed)
    if sampling_rate is not None:
        df[time_key] = df[time_key].round(sampling_rate)

    # Pivot table (group by time_key + facets, sum values)
    grouped = df.pivot_table(
        index=time_key,
        columns=facets,
        values=values,
        aggfunc='sum',
        fill_value=fill_value,
    )  # (T, d1*d2*...*dM)

    # Reindex to ensure all combinations of facet levels are present
    # Example:
    #   facets = ['store', 'item']
    #   store levels = ['SuperMart', 'DrugStore'], item levels = ['Rice', 'Bread', 'Milk']
    #   facet_levels = [['SuperMart', 'DrugStore'], ['Rice', 'Bread', 'Milk']]
    #   grouped.columns = MultiIndex([('SuperMart', 'Rice'), ('SuperMart', 'Bread'), ...])
    #   d_sizes = [2, 3]
    facet_levels = [grouped.columns.levels[i].tolist() for i in range(grouped.columns.nlevels)]
    full_cols = pd.MultiIndex.from_product(facet_levels, names=facets)
    grouped = grouped.reindex(columns=full_cols, fill_value=fill_value)
    d_sizes = [len(level) for level in facet_levels]
    n_cols = grouped.shape[1]
    if n_cols != int(np.prod(d_sizes)):
        raise ValueError(
            f'product(d_sizes)={np.prod(d_sizes)} but pivot columns={n_cols}; '
            f'facet_levels may not match observed categories.'
        )

    # Tensorization
    tts = np.asarray(grouped.values)  # (T, d1*d2*...*dM)
    tts = np.reshape(tts, (-1, *d_sizes))  # (T, d1, d2, ..., dM)

    return tts, grouped.index, grouped.columns


def tts2df(
    tensor: np.ndarray,
    time_key: str,
    facets: List[str],
    values: str = 'values',
    time_index: Optional[Sequence] = None,
    facet_levels: Optional[List[Sequence]] = None,
    columns: Optional[pd.MultiIndex] = None,
    long: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Reconstruct a pd.DataFrame from a tensor time series.

    Parameters
    ----------
    tensor : np.ndarray
        M+1-th order tensor of shape (T, d1, d2, ..., dM).
    time_key : str
        Column name for time in the output DataFrame.
    facets : list of str
        Facet (mode) names in the same order used in df2tts (columns order).
    values : str, default "values"
        Value column name for long-form output.
    time_index : optional sequence-like, length n
        Timestamps after rounding/resampling and date filtering.
        Required if `columns` is not provided.
    facet_levels : optional list of sequences
        List `[L1, ..., LM]` where `len(Lm) == dm`. Required if `columns` is not provided.
    columns : optional pandas.MultiIndex
        Exact MultiIndex columns used in the pivot table inside df2tts.
        If provided, `facet_levels` is ignored.
    long : bool, default True
        If True, return long-form DataFrame: [time_key] + facets + [values].
        If False, return wide (mode-0 unfolding): index=time, columns=MultiIndex(facets).

    Returns
    -------
    pd.DataFrame
        Reconstructed DataFrame (long or wide).
    """
    if tensor.ndim < 2:
        raise ValueError('`tensor` must have at least 2 dimensions: (n, d1, ..., dM).')

    n = tensor.shape[0]
    d_sizes = list(tensor.shape[1:])
    M = len(d_sizes)

    if len(facets) != M:
        raise ValueError(f'`facets` length ({len(facets)}) must equal tensor order M ({M}).')

    # ---- Construct columns MultiIndex ----
    if columns is not None:
        # Format B: use the provided MultiIndex as-is
        if not isinstance(columns, pd.MultiIndex):
            raise TypeError('`columns` must be a pandas.MultiIndex.')
        if columns.nlevels != M:
            raise ValueError('`columns` must have exactly M levels (one per facet).')
        if columns.names is None or any(name is None for name in columns.names):
            columns = columns.set_names(facets)
        flat_cols = columns
    else:
        # Format A: construct MultiIndex from facet_levels
        if time_index is None or facet_levels is None:
            raise ValueError('`time_index` and `facet_levels` are required if `columns` is not provided.')
        if len(facet_levels) != M:
            raise ValueError('`facet_levels` must have M sequences (one per facet).')
        # Check lengths
        for k, (levs, d) in enumerate(zip(facet_levels, d_sizes)):
            if len(levs) != d:
                raise ValueError(f'facet_levels[{k}] length ({len(levs)}) != tensor.shape[{k + 1}] ({d}).')

        tuples = list(product(*facet_levels))
        flat_cols = pd.MultiIndex.from_tuples(tuples, names=facets)

    # ---- Construct wide DataFrame ----
    wide = tensor.reshape(n, -1)
    if time_index is None:
        time_index = pd.RangeIndex(n, name=time_key)
    else:
        if len(time_index) != n:
            raise ValueError(f'`time_index` length ({len(time_index)}) != n ({n}).')

    df_wide = pd.DataFrame(wide, index=pd.Index(time_index, name=time_key), columns=flat_cols)

    if not long:
        return df_wide

    # ---- Transform to long format ----
    # index: time_key, columns: MultiIndex(facets) -> long: [time_key] + facets + [values]
    df_long = (
        df_wide.stack(level=list(range(M)), future_stack=True)  # stack all facet levels
        .rename(values)
        .reset_index()
    )

    if dropna:
        df_long = df_long.dropna(subset=[values])
        vals = df_long[values]
        df_long[values] = vals.astype(object)
        df_long.loc[pd.isna(df_long[values]), values] = None

    # Organize columns as [time_key] + facets + [values]
    return df_long[[time_key] + facets + [values]]


if __name__ == '__main__':
    rng = pd.date_range('2025-01-01', periods=3, freq='D')
    stores = ['SuperMart', 'DrugStore']
    items = ['Rice', 'Bread', 'Milk']
    data = [
        # t=0
        (rng[0], 'SuperMart', 'Rice', 5.0),
        (rng[0], 'SuperMart', 'Bread', 2.0),
        (rng[0], 'DrugStore', 'Rice', 3.0),
        # t=1
        (rng[1], 'SuperMart', 'Rice', 7.0),
        (rng[1], 'DrugStore', 'Rice', 0.0),
        # t=2
        (rng[2], 'SuperMart', 'Bread', 4.0),
        (rng[2], 'DrugStore', 'Milk', 1.0),
    ]
    df = pd.DataFrame(data, columns=['time', 'store', 'item', 'values'])
    keys = df.columns.to_list()
    keys.remove('values')

    tensor, time_index, columns = df2tts(
        df=df,
        time_key='time',
        facets=['store', 'item'],
        values='values',
        sampling_rate='D',
        fill_value=None,
    )

    long_rec = tts2df(
        tensor,
        time_key='time',
        facets=['store', 'item'],
        values='values',
        time_index=time_index,
        columns=columns,
        long=True,
        dropna=True,
    )

    wide_rec = tts2df(
        tensor,
        time_key='time',
        facets=['store', 'item'],
        values='values',
        time_index=time_index,
        columns=columns,
        long=False,
        dropna=True,
    )

    df = df.sort_values(by=keys).reset_index(drop=True)
    long_rec = long_rec.sort_values(by=keys).reset_index(drop=True)

    print('\n=== Original DataFrame ===')
    print(df)
    print('\n=== Transformed Tensor ===')
    print(tensor)
    print('\n=== Reconstructed Long-form DataFrame ===')
    print(tabulate(long_rec, headers='keys', tablefmt='fancy_grid', showindex=False))
    print('\n=== Reconstructed Wide-form DataFrame ===')
    print(tabulate_wide(wide_rec, style='multirow', tablefmt='fancy_grid'))

    pdt.assert_frame_equal(
        df,
        long_rec,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
        check_dtype=False,
    )
