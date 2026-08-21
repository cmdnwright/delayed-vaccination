from typing import Callable, Mapping
import numpy as np

from scipy.interpolate import PchipInterpolator
from src.utils import load_config

config = load_config('config.yaml')

TODDLER_DOSE_CHANNEL = 1
SCHOOL_ENTRY_CHANNEL = config['age_structure']['n_vaccination_rates'] - 1

# historical vaccine coverage at documented years to create historically informed 
# vaccination estimates for validation window
HISTORICAL_COVERAGE_CHECKPOINTS: dict[float, tuple[float, float]] = {
    1965.0: (0.00, 0.00),
    1966.0: (0.15, 0.25),
    1970.0: (0.40, 0.60),
    1974.0: (0.35, 0.55),
    1978.0: (0.45, 0.65),
    1981.0: (0.65, 0.93),
    1985.0: (0.53, 0.85),
    1990.0: (0.48, 0.80),
    1994.0: (0.80, 0.90),
    1996.0: (0.90, 0.95),
    2000.0: (0.92, 0.97),
}


def build_extended_vaccination_rate_functions(zero_before_year: float = 1965.0, checkpoints_calendar_year: Mapping[float, tuple[float, float]] = HISTORICAL_COVERAGE_CHECKPOINTS) -> Callable[[float], np.ndarray]:
    '''builds time dependent vaccination rate function using PCHIP interpolation for shape preservation and monotonicity

    Parameters
    ----------
    zero_before_year : float
        calendar year at and before which coverage is forced to exactly
        zero, matching the calibration model's zero-vaccination
        assumption through its fitting window.
    checkpoints_calendar_year : Mapping[float, tuple[float, float]]
        {calendar_year: (toddler_coverage, school_entry_coverage)}
        checkpoints, by default `HISTORICAL_COVERAGE_CHECKPOINTS`.

    Returns
    -------
    Callable[[float], np.ndarray]
        calendar-year -> vaccination-rate-vector function.

    Raises
    ------
    ValueError
        if `checkpoints_calendar_year` is empty, if any checkpoint year
        precedes `zero_before_year`, or if any coverage value falls
        outside [0, 1].
    '''
    if not checkpoints_calendar_year:
        raise ValueError('checkpoints_calendar_year must not be empty')

    checkpoints = {
        float(year): (float(values[0]), float(values[1]))
        for year, values in checkpoints_calendar_year.items()
    }
    years = np.array(sorted(checkpoints), dtype=float)
    if years[0] < zero_before_year:
        raise ValueError(
            'checkpoint years cannot precede zero_before_year; '
            'the calibration-era zero-vaccination constraint would be ambiguous'
        )

    toddler = np.array([checkpoints[y][0] for y in years], dtype=float)
    school = np.array([checkpoints[y][1] for y in years], dtype=float)

    for name, values in (('toddler', toddler), ('school-entry', school)):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f'{name} coverage must lie in [0, 1]')

    toddler_interp = PchipInterpolator(years, toddler, extrapolate=False)
    school_interp = PchipInterpolator(years, school, extrapolate=False)
    y_min, y_max = years[0], years[-1]

    def vaccination_rate_fn(calendar_year: float) -> np.ndarray:
        v = np.zeros(config['age_structure']['n_vaccination_rates'], dtype=float)
        if calendar_year <= zero_before_year:
            return v

        clamped = min(max(float(calendar_year), y_min), y_max)
        v[TODDLER_DOSE_CHANNEL] = float(toddler_interp(clamped))
        v[SCHOOL_ENTRY_CHANNEL] = float(school_interp(clamped))
        return v

    return vaccination_rate_fn


def build_vaccination_rate_functions(validation_start_year: float) -> Callable[[float], np.ndarray]:
    '''
    Parameters
    ----------
    validation_start_year : float
        calendar year at and before which vaccination is forced to zero,
        passed through as `zero_before_year`.

    Returns
    -------
    Callable[[float], np.ndarray]
        calendar-year -> vaccination-rate-vector function.
    '''
    return build_extended_vaccination_rate_functions(
        zero_before_year=validation_start_year,
        checkpoints_calendar_year=HISTORICAL_COVERAGE_CHECKPOINTS,
    )