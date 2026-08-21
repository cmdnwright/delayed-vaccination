from typing import Any, Dict, Optional, Sequence
import numpy as np

from src.utils import load_config

config = load_config('config.yaml')


def randomized_pit(case_draws: np.ndarray, observed_case_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''computes randomized probability integral transform (PIT) for discrete count data

    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    rng : np.random.Generator
        random number generator for diagnostic randomization.

    Returns
    -------
    np.ndarray
        randomized PIT values with shape (T-1,).
    '''
    P, T_minus_1 = case_draws.shape
    pit = np.empty(T_minus_1)
    for k in range(T_minus_1):
        draws = case_draws[:, k]
        y = observed_case_counts[k]
        f_lower = np.mean(draws < y)
        f_upper = np.mean(draws <= y)
        v = rng.uniform()
        pit[k] = f_lower + v * (f_upper - f_lower)
    return pit


def pit_uniformity_test(pit_values: np.ndarray) -> Dict[str, Any]:
    '''one sample Kolmogorov-Smirnov test comparing pit_values against Uniform(0,1)

    Parameters
    ----------
    pit_values : np.ndarray
        array of probability integral transform values.

    Returns
    -------
    Dict[str, Any]
        dictionary containing KS statistic, p-value, and sample size.
    '''
    from scipy import stats

    ks_stat, p_value = stats.kstest(pit_values, 'uniform')
    return {'ks_statistic': float(ks_stat), 'p_value': float(p_value), 'n': int(len(pit_values))}


def interval_coverage(case_draws: np.ndarray, observed_case_counts: np.ndarray, levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Dict[float, float]:
    '''calculates empirical coverage of central prediction intervals across specified nominal levels

    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    levels : Sequence[float], default=(0.5, 0.8, 0.9, 0.95)
        nominal coverage levels to evaluate.

    Returns
    -------
    Dict[float, float]
        mapping from nominal coverage level to empirical coverage fraction.
    '''
    out = {}
    for level in levels:
        lo_q = (1 - level) / 2 * 100
        hi_q = (1 + level) / 2 * 100
        lo, hi = np.percentile(case_draws, [lo_q, hi_q], axis=0)
        in_band = (observed_case_counts >= lo) & (observed_case_counts <= hi)
        out[float(level)] = float(in_band.mean())
    return out


def weighted_interval_score(case_draws: np.ndarray, observed_case_counts: np.ndarray, levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Dict[str, Any]:
    '''compute Weighted Interval Score (WIS) as in Bracher et al. (2021).
    
    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    levels : Sequence[float], default=(0.5, 0.8, 0.9, 0.95)
        nominal interval levels used for scoring.

    Returns
    -------
    Dict[str, Any]
        dictionary containing mean WIS, interval-wise scores, and component decompositions.
    '''
    T_minus_1 = case_draws.shape[1]
    median = np.median(case_draws, axis=0)
    K = len(levels)
    total = np.abs(observed_case_counts - median) * 0.5
    sharpness_total = np.zeros(T_minus_1)
    under_total = np.zeros(T_minus_1)
    over_total = np.zeros(T_minus_1)
    for level in levels:
        alpha = 1 - level
        lo, hi = np.percentile(case_draws, [alpha / 2 * 100, (1 - alpha / 2) * 100], axis=0)
        width = hi - lo
        under = np.maximum(lo - observed_case_counts, 0)
        over = np.maximum(observed_case_counts - hi, 0)
        interval_score = width + (2 / alpha) * under + (2 / alpha) * over
        weight = alpha / 2
        total += weight * interval_score
        sharpness_total += weight * width
        under_total += weight * (2 / alpha) * under
        over_total += weight * (2 / alpha) * over
    denom = K + 0.5
    wis_per_interval = total / denom
    return {
        'wis_mean': float(wis_per_interval.mean()),
        'wis_per_interval': wis_per_interval.tolist(),
        'sharpness_component_mean': float((sharpness_total / denom).mean()),
        'underprediction_component_mean': float((under_total / denom).mean()),
        'overprediction_component_mean': float((over_total / denom).mean()),
        'levels': list(levels),
    }


def crps_ensemble(case_draws: np.ndarray, observed_case_counts: np.ndarray, max_particles_for_pairs: int = 500, seed: int = 0) -> Dict[str, Any]:
    '''estimates Continuous Ranked Probability Score (CRPS) using the unbiased pairwise difference ensemble estimator (Gneiting & Raftery 2007).

    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    max_particles_for_pairs : int, default=500
        maximum particle ensemble size used for pairwise difference calculations.
    seed : int, default=0
        random seed for particle subsampling.

    Returns
    -------
    Dict[str, Any]
        dictionary containing mean CRPS and per-interval CRPS values.
    '''
    P, T_minus_1 = case_draws.shape
    n_sub = min(P, max_particles_for_pairs)
    if n_sub < P:
        idx = np.random.default_rng(seed).choice(P, n_sub, replace=False)
        sub = case_draws[idx]
    else:
        sub = case_draws
    crps = np.empty(T_minus_1)
    for k in range(T_minus_1):
        x = sub[:, k]
        term1 = np.mean(np.abs(x - observed_case_counts[k]))
        term2 = 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
        crps[k] = term1 - term2
    return {'crps_mean': float(crps.mean()), 'crps_per_interval': crps.tolist()}


def peak_timing_distribution(case_draws: np.ndarray, validation_times: np.ndarray, observed_case_counts: np.ndarray, tolerance_years: float = 0.25) -> Dict[str, Any]:
    '''calculates per particle peak timing and evaluates probability of peaking within tolerance_years of observed peak.
    
    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    validation_times : np.ndarray
        elapsed observation time points with shape (T,).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    tolerance_years : float, default=0.25
        time threshold in years for evaluating peak coincidence.

    Returns
    -------
    Dict[str, Any]
        dictionary with peak timing metrics, probabilities, and summary statistics.
    '''
    obs_peak_idx = int(np.argmax(observed_case_counts))
    obs_peak_time = float(validation_times[obs_peak_idx])
    particle_peak_idx = np.argmax(case_draws, axis=1)  # (P,)
    interval_times = validation_times[:-1]
    particle_peak_time = interval_times[particle_peak_idx]
    within_tol = np.abs(particle_peak_time - obs_peak_time) <= tolerance_years
    return {
        'observed_peak_time': obs_peak_time,
        'observed_peak_cases': float(observed_case_counts[obs_peak_idx]),
        'particle_peak_times': particle_peak_time.tolist(),
        'prob_peak_within_tolerance': float(within_tol.mean()),
        'tolerance_years': tolerance_years,
        'median_particle_peak_time': float(np.median(particle_peak_time)),
        'iqr_particle_peak_time': [
            float(np.percentile(particle_peak_time, 25)),
            float(np.percentile(particle_peak_time, 75)),
        ],
    }


def age_stratified_incidence_summary(simulated_incidence: np.ndarray, age_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    '''summarizes posterior-predictive mean and 5-95% intervals for age-stratified underlying incidence
    
    Parameters
    ----------
    simulated_incidence : np.ndarray
        simulated underlying $E \to I$ incidence array with shape (P, T-1, A).
    age_labels : Optional[Sequence[str]], default=None
        labels for age group categories.

    Returns
    -------
    Dict[str, Any]
        dictionary containing posterior mean incidence, 5-95% confidence intervals, and age share breakdown.
    '''
    if age_labels is None:
        age_labels = config['age_structure']['age_group_labels']
    mean = simulated_incidence.mean(axis=0)  # (T-1, A)
    lo, hi = np.percentile(simulated_incidence, [5, 95], axis=0)  # (T-1, A) each
    age_share_of_total = mean.sum(axis=0) / max(mean.sum(), 1e-12)  # (A,)
    return {
        'age_labels': list(age_labels) if age_labels else None,
        'mean': mean.tolist(),  # (T-1, A)
        'ci_5_95': np.stack([lo, hi], axis=-1).tolist(),  # (T-1, A, 2)
        'age_share_of_total_incidence': age_share_of_total.tolist(),  # (A,)
    }


def compute_all_metrics(case_draws: np.ndarray, validation_times: np.ndarray, observed_case_counts: np.ndarray, simulated_incidence: np.ndarray, pit_rng: np.random.Generator, coverage_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95), peak_tolerance_years: float = 0.25, age_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    '''run full suite of posterior-predictive validation diagnostics
    
    Parameters
    ----------
    case_draws : np.ndarray
        posterior-predictive reported-case draws with shape (P, T-1).
    validation_times : np.ndarray
        elapsed time points array with shape (T,).
    observed_case_counts : np.ndarray
        observed case counts with shape (T-1,).
    simulated_incidence : np.ndarray
        simulated incidence array with shape (P, T-1, A).
    pit_rng : np.random.Generator
        random generator for PIT randomization.
    coverage_levels : Sequence[float], default=(0.5, 0.8, 0.9, 0.95)
        nominal interval coverage levels.
    peak_tolerance_years : float, default=0.25
        tolerance window in years for peak timing analysis.
    age_labels : Optional[Sequence[str]], default=None
        labels corresponding to age groups.

    Returns
    -------
    Dict[str, Any]
        dictionary containing all computed validation metrics ready for serialization.
    '''
    pit_values = randomized_pit(case_draws, observed_case_counts, pit_rng)
    return {
        'pit_values': pit_values.tolist(),
        'pit_uniformity_test': pit_uniformity_test(pit_values),
        'interval_coverage': interval_coverage(case_draws, observed_case_counts, coverage_levels),
        'weighted_interval_score': weighted_interval_score(case_draws, observed_case_counts, coverage_levels),
        'crps': crps_ensemble(case_draws, observed_case_counts),
        'peak_timing': peak_timing_distribution(
            case_draws, validation_times, observed_case_counts, peak_tolerance_years
        ),
        'age_stratified_incidence': age_stratified_incidence_summary(simulated_incidence, age_labels),
    }