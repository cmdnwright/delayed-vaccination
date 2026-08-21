from typing import Callable, Optional
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.data import contact_matrix
from src.utils import load_config

config = load_config('config.yaml')

# decennial Texas census totals
TEXAS_CENSUS_POPULATION: dict[int, int] = {
    1920: 4_663_228,
    1930: 5_824_715,
    1940: 6_414_824,
    1950: 7_711_194,
    1960: 9_579_677,
    1970: 11_196_730,
    1980: 14_229_191,
    1990: 16_986_510,
    2000: 20_851_820,
    2010: 25_145_561,
    2020: 29_145_505,
}

# Documented crude birth/death rate data points per 1,000 population for the given calendar year
# sourced for documentation purposes
VITAL_RATE_DATA_POINTS: dict[int, dict[str, object]] = {
    1950: {
        'birth_rate_per_1000': 24.1,
        'death_rate_per_1000': 9.6,
        'source': (
            'US national CBR/CDR, Census Bureau vital-rate table for '
            'census years 1920-1990 '
        ),
    },
    1960: {
        'birth_rate_per_1000': 23.7,
        'death_rate_per_1000': 9.4,
        'source': (
            'US national CBR/CDR, same Census Bureau table as 1950 entry '
        ),
    },
    2012: {
        'birth_rate_per_1000': 14.7,
        'death_rate_per_1000': 6.7,
        'source': 'Texas DSHS, Summary of Vital Statistics for Texas 2012',
    },
    2021: {
        'birth_rate_per_1000': 12.7,
        'death_rate_per_1000': 9.1,
        'source': (
            'Texas DSHS resident births (373,594) and deaths (267,651) '
            'for 2021, divided by 29.5M estimated TX population '
        ),
    },
}


def population_for_year(year: int, census: dict[int, int] = TEXAS_CENSUS_POPULATION) -> float:
    '''estimates texas population per year using log linearly interpolated points between nearest
    census estimates. uses log linear since population growth is compounding

    Parameters
    ----------
    year : int
        any year, in or out of [1920, 2020]. Years outside that range
        extrapolate off the nearest decade's growth rate.
    census : dict[int, int]
        decennial population totals keyed by year, by default
        `TEXAS_CENSUS_POPULATION`.

    Returns
    -------
    float
        estimated Texas population at `year`.
    '''
    years = sorted(census)
    if year in census:
        return float(census[year])

    if year < years[0]:
        y0, y1 = years[0], years[1]
    elif year > years[-1]:
        y0, y1 = years[-2], years[-1]
    else:
        y0 = max(y for y in years if y < year)
        y1 = min(y for y in years if y > year)

    p0, p1 = census[y0], census[y1]
    r = np.log(p1 / p0) / (y1 - y0)
    return float(p0 * np.exp(r * (year - y0)))


def build_vital_rate_functions(extra_data_points: Optional[dict[int, dict[str, object]]] = None, base_data_points: dict[int, dict[str, object]] = VITAL_RATE_DATA_POINTS) -> tuple[Callable[[float], float], Callable[[float], float]]:
    '''builds time varying birth and death rate functions rather than cross interval estimates. uses PCHIP interpolation
    to prevent overshooting and preserve monotonicity

    Parameters
    ----------
    extra_data_points : dict, optional
        additional {year: {'birth_rate_per_1000': ..., 'death_rate_per_1000': ...}}
        entries merged on top of `base_data_points` (overriding any
        matching year). Use this to add project-window-specific
        measurements or externally-sourced future projections without
        editing the module-level table.
    base_data_points : dict
        defaults to `VITAL_RATE_DATA_POINTS`.

    Returns
    -------
    birth_rate_fn, death_rate_fn : Callable[[float], float]
        each takes a calendar year (float, e.g. 1953.5) and returns an
        annualized rate (fraction per year, already divided by 1000).
        suitable to pass directly as parameters['birth_rate'] and
        parameters['death_rate'] which must be called
        as birth_rate(calendar_year) when they're callable rather
        than treating them as a bare float.

    Raises
    ------
    valueError
        if fewer than 2 data points are available to interpolate from.
    '''
    points = dict(base_data_points)
    if extra_data_points:
        points.update(extra_data_points)
    if len(points) < 2:
        raise ValueError(
            f'Need at least 2 vital-rate data points to build an interpolant, got {len(points)}'
        )

    years = np.array(sorted(points), dtype=float)
    birth = np.array([points[int(y)]['birth_rate_per_1000'] for y in years], dtype=float) / 1000.0
    death = np.array([points[int(y)]['death_rate_per_1000'] for y in years], dtype=float) / 1000.0

    birth_interp = PchipInterpolator(years, birth, extrapolate=False)
    death_interp = PchipInterpolator(years, death, extrapolate=False)
    y_min, y_max = years[0], years[-1]

    def birth_rate_fn(calendar_year: float) -> float:
        clamped = min(max(calendar_year, y_min), y_max)
        return float(birth_interp(clamped))

    def death_rate_fn(calendar_year: float) -> float:
        clamped = min(max(calendar_year, y_min), y_max)
        return float(death_interp(clamped))

    return birth_rate_fn, death_rate_fn