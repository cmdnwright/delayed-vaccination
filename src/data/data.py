from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import load_config

config = load_config('config.yaml')


def load_raw_tycho(path: Path = config['paths']['raw_tycho_path']) -> pd.DataFrame:
    '''loads raw case count data

    Parameters
    ----------
    path : Path
        location of the raw Tycho CSV export.

    Returns
    -------
    pd.DataFrame
        raw, unfiltered Tycho records with parsed period dates.
    '''
    raw = pd.read_csv(
        path,
        parse_dates=['PeriodStartDate', 'PeriodEndDate'],
        low_memory=False,
    )
    return raw


def filter_texas_measles(raw: pd.DataFrame, cfg=config) -> pd.DataFrame:
    '''filters raw data for just measles case reports. checks against cumulative series reporting common in
    tycho data

    Parameters
    ----------
    raw : pd.DataFrame
        output of `load_raw_tycho`.
    cfg : module
        config module (or object exposing the same attributes) supplying
        the condition/country/admin1 filter values.

    Returns
    -------
    pd.DataFrame
        filtered records, one row per reporting period/location/subgroup.
    '''
    is_measles = raw['ConditionName'].str.upper() == cfg['data']['tycho_condition_name']
    is_us = raw['CountryISO'].str.upper() == cfg['data']['tycho_country_iso']
    is_texas = raw['Admin1Name'].str.upper() == cfg['data']['tycho_admin1_name']

    cumulative_flag = raw['PartOfCumulativeCountSeries']
    if cumulative_flag.dtype == object:
        cumulative_flag = cumulative_flag.astype(str).str.upper().eq('TRUE')
    is_incident = ~cumulative_flag.astype(bool)

    filtered = raw.loc[is_measles & is_us & is_texas & is_incident].copy()
    return filtered


def clean_and_aggregate_series(filtered: pd.DataFrame, resolution: str = config['data']['temporal_resolution']) -> pd.DataFrame:
    '''cleans bad data values and aggregates higher resolution data breakdowns into full state level reporting

    Parameters
    ----------
    filtered : pd.DataFrame
        output of `filter_texas_measles`.
    resolution : str
        pandas offset alias to resample to (default weekly, matching the
        native Tycho reporting cadence for this range).

    Returns
    -------
    pd.DataFrame
        Columns: `period_start`, `case_count`, aggregated across all
        Admin2/City/age/subpopulation breakdowns within Texas, indexed to
        a complete, gap-free date range with unreported weeks filled as
        zero cases.
    '''
    df = filtered.copy()
    df = df.dropna(subset=['PeriodStartDate', 'CountValue'])
    df['CountValue'] = pd.to_numeric(df['CountValue'], errors='coerce')
    df = df.dropna(subset=['CountValue'])

    # sum across all breakdowns
    grouped = (
        df.groupby(pd.Grouper(key='PeriodStartDate', freq=resolution))['CountValue']
        .sum()
        .rename('case_count')
        .reset_index()
        .rename(columns={'PeriodStartDate': 'period_start'})
    )

    # reindex to a complete date range so gaps are zeros
    full_range = pd.date_range(
        grouped['period_start'].min(), grouped['period_start'].max(), freq=resolution
    )
    grouped = (
        grouped.set_index('period_start')
        .reindex(full_range)
        .rename_axis('period_start')
        .reset_index()
    )
    grouped['case_count'] = grouped['case_count'].fillna(0.0)

    return grouped


def build_texas_measles_series(cfg=config) -> None:
    '''runs the full case processing pipeline and saves the result to data/processed/

    Parameters
    ----------
    cfg : module
        config module (or object exposing the same attributes).

    Returns
    -------
    none
    '''
    raw = load_raw_tycho(cfg['paths']['raw_tycho_path'])
    filtered = filter_texas_measles(raw, cfg)
    series = clean_and_aggregate_series(filtered, cfg['data']['temporal_resolution'])

    Path(cfg['paths']['data_processed_dir']).mkdir(parents=True, exist_ok=True)
    series.to_csv(cfg['paths']['processed_cases_path'], index=False)


def series_to_elapsed_years(series: pd.DataFrame) -> np.ndarray:
    '''converts a processed case series dates to elapsed years since start

    Parameters
    ----------
    series : pd.DataFrame
        output of `load_processed_series` / `clean_and_aggregate_series`;
        must have a `period_start` column of datetime64 values.

    Returns
    -------
    np.ndarray, shape (len(series),)
        elapsed time in years since `series['period_start'].min()`, i.e.
        `t[0] == 0.0`.
    '''
    elapsed_days = (series['period_start'] - series['period_start'].min()).dt.days
    return elapsed_days.to_numpy(dtype=float) / 365.25


def load_processed_series(cfg=config) -> pd.DataFrame:
    '''loads a processed series

    Parameters
    ----------
    cfg : module
        config module (or object exposing the same attributes).

    Returns
    -------
    pd.DataFrame
        Columns: `period_start` (datetime64), `case_count` (float).
    '''
    return pd.read_csv(cfg['paths']['processed_cases_path'], parse_dates=['period_start'])