from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from src.experiments import vaccination
from src.model import observation_model, particle_filter, stochastic_model
from src.utils import load_config

config = load_config("config.yaml")


def capture_calibration_final_cloud(theta_hat: Dict[str, Any], observed_times: np.ndarray, observed_case_counts: np.ndarray, fixed_parameters: Dict[str, Any], fit_window_start_year: float, n_particles: int = config["calibration"]["particle_filter_n_particles_final"], seed: int = 1, score_start: Optional[float] = None) -> particle_filter.ParticleFilterResult:
    '''re-run particle filter once at theta_hat over calibration window to capture final_states
    
    Parameters
    ----------
    theta_hat : Dict[str, Any]
        calibrated parameter point estimate dictionary.
    observed_times : np.ndarray
        calibration window observation times.
    observed_case_counts : np.ndarray
        calibration window observed case series.
    fixed_parameters : Dict[str, Any]
        fixed epidemiological and demographic model parameters.
    fit_window_start_year : float
        calendar start year for calibration fitting window.
    n_particles : int, default=config["calibration"]["particle_filter_n_particles_final"]
        number of particles in particle cloud.
    seed : int, default=1
        random seed for particle filter run.
    score_start : Optional[float], default=None
        start time for scoring evaluation.

    Returns
    -------
    particle_filter.ParticleFilterResult
        particle filter result object containing final particle states.
    '''
    rng = np.random.default_rng(seed)
    return particle_filter.run_particle_filter(
        theta=theta_hat,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        n_particles=n_particles,
        rng=rng,
        fit_window_start_year=fit_window_start_year,
        score_start=score_start,
    )


def build_validation_fixed_parameters(calibration_fixed_parameters: Dict[str, Any], validation_start_year: float) -> Dict[str, Any]:
    '''construct validation fixed parameters from calibration baseline
    
    Parameters
    ----------
    calibration_fixed_parameters : Dict[str, Any]
        fixed parameters used during model calibration.
    validation_start_year : float
        calendar start year of out-of-sample validation period.

    Returns
    -------
    Dict[str, Any]
        updated fixed parameters dictionary with time-varying vaccination functions.
    '''
    validation_fixed = dict(calibration_fixed_parameters)
    validation_fixed["vaccination_rates"] = vaccination.build_vaccination_rate_functions(
        validation_start_year=validation_start_year,
    )
    return validation_fixed


@dataclass
class ValidationResult:
    '''out of sample posterier predictive validation outputs
    
    Parameters
    ----------
    validation_times : np.ndarray
        elapsed validation observation times with shape (T,).
    simulated_incidence : np.ndarray
        per-particle and per-interval incidence array with shape (P, T-1, A).
    simulated_case_counts_mean : np.ndarray
        posterior-predictive reported-case mean with shape (T-1,).
    simulated_case_counts_ci : np.ndarray
        posterior-predictive 5th/95th percentile bounds with shape (T-1, 2).
    simulated_case_draws : np.ndarray
        full per-particle reported-case draws with shape (P, T-1).
    observed_case_counts : np.ndarray
        held-out target case counts with shape (T-1,).
    log_predictive_density : float
        summed log predictive density across validation intervals.
    n_particles : int
        number of simulation particles.
    final_particle_cloud : Optional[stochastic_model.StochasticParticleCloud], default=None
        final particle cloud state at end of validation window.
    '''

    validation_times: np.ndarray
    simulated_incidence: np.ndarray
    simulated_case_counts_mean: np.ndarray
    simulated_case_counts_ci: np.ndarray
    simulated_case_draws: np.ndarray
    observed_case_counts: np.ndarray
    log_predictive_density: float
    n_particles: int
    final_particle_cloud: Optional[stochastic_model.StochasticParticleCloud] = None


def run_validation(theta_hat: Dict[str, Any], calibration_final_cloud: stochastic_model.StochasticParticleCloud, validation_observed_times: np.ndarray, validation_observed_case_counts: np.ndarray, validation_fixed_parameters: Dict[str, Any], validation_start_year: float, rng: np.random.Generator, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config["model"]["default_importation_rate"], dtype=float), return_final_cloud: bool = False) -> ValidationResult:
    '''forward simulate model across validation window without particle filtering reweighting or resampling
    
    Parameters
    ----------
    theta_hat : Dict[str, Any]
        calibrated parameter point estimate dictionary.
    calibration_final_cloud : stochastic_model.StochasticParticleCloud
        final particle cloud from calibration window run.
    validation_observed_times : np.ndarray
        elapsed validation time points array with shape (T,).
    validation_observed_case_counts : np.ndarray
        held-out observed case counts series with shape (T,) or (T-1,).
    validation_fixed_parameters : Dict[str, Any]
        fixed parameters dictionary containing time-varying vaccination rates.
    validation_start_year : float
        calendar start year of validation period.
    rng : np.random.Generator
        random number generator for stochastic simulation.
    dt : Optional[float], default=None
        simulation integration time step in years.
    importation_rate : np.ndarray, default=config["model"]["default_importation_rate"]
        age-stratified viral importation rates.
    return_final_cloud : bool, default=False
        whether to attach final particle cloud state to returned result.

    Returns
    -------
    ValidationResult
        dataclass holding predictive draws, summaries, and predictive density score.

    Raises
    ------
    ValueError
        if `validation_observed_case_counts` length does not match observation time array.
    '''
    cloud = stochastic_model.StochasticParticleCloud(
        t=0.0, counts=calibration_final_cloud.counts.copy()
    )
    P = cloud.n_particles
    A = config["age_structure"]["n_age_groups"]

    birth_rate = particle_filter._wrap_calendar_rate(
        validation_fixed_parameters["birth_rate"], validation_start_year
    )
    death_rate = particle_filter._wrap_calendar_rate(
        validation_fixed_parameters["death_rate"], validation_start_year
    )
    vaccination_rates = particle_filter._wrap_calendar_rate(
        validation_fixed_parameters["vaccination_rates"], validation_start_year
    )

    parameters = {
        "beta_0": theta_hat["beta_0"],
        "beta_1": theta_hat["beta_1"],
        "phi": theta_hat["phi"],
        "gamma": validation_fixed_parameters["gamma"],
        "sigma": validation_fixed_parameters["sigma"],
        "maternal_waning_rate": validation_fixed_parameters["maternal_waning_rate"],
        "birth_rate": birth_rate,
        "death_rate": death_rate,
        "vaccination_rates": vaccination_rates,
        "contact_matrix": validation_fixed_parameters["contact_matrix"],
    }
    rho = theta_hat["rho"]
    phi_obs = theta_hat["phi_obs"]

    T = len(validation_observed_times)
    if len(validation_observed_case_counts) != T - 1 and len(validation_observed_case_counts) != T:
        raise ValueError(
            "validation_observed_case_counts must have length T (aligned to observation "
            "times, first entry unused) or T-1 (aligned to intervals)"
        )
    y = (
        validation_observed_case_counts[1:]
        if len(validation_observed_case_counts) == T
        else validation_observed_case_counts
    )

    simulated_incidence = np.zeros((P, T - 1, A), dtype=np.int64)
    for k in range(1, T):
        duration = float(validation_observed_times[k] - validation_observed_times[k - 1])
        cloud, inc = stochastic_model.simulate_interval_batch(
            cloud, duration, parameters, rng, dt=dt, importation_rate=importation_rate
        )
        simulated_incidence[:, k - 1, :] = inc

    total_incidence = simulated_incidence.sum(axis=2)  # (P, T-1)
    mu = rho * total_incidence  # (P, T-1)

    case_draws = np.stack(
        [observation_model.negbin_rvs(mu[:, k], phi_obs, rng) for k in range(T - 1)],
        axis=1,
    )  # (P, T-1)
    ci_low, ci_high = np.percentile(case_draws, [5, 95], axis=0)

    logpmf_per_particle = np.stack(
        [observation_model.negbin_logpmf(np.full(P, y[k]), mu[:, k], phi_obs) for k in range(T - 1)],
        axis=1,
    )  # (P, T-1)
    max_lp = np.max(logpmf_per_particle, axis=0)
    with np.errstate(invalid="ignore"):
        mean_density = np.where(
            np.isfinite(max_lp),
            np.exp(logpmf_per_particle - np.where(np.isfinite(max_lp), max_lp, 0.0)[None, :]).mean(axis=0),
            0.0,
        )
    with np.errstate(divide="ignore"):
        log_density_per_t = np.where(mean_density > 0, max_lp + np.log(mean_density), -np.inf)
    log_predictive_density = float(np.sum(log_density_per_t))

    return ValidationResult(
        validation_times=validation_observed_times,
        simulated_incidence=simulated_incidence,
        simulated_case_counts_mean=case_draws.mean(axis=0),
        simulated_case_counts_ci=np.stack([ci_low, ci_high], axis=1),
        simulated_case_draws=case_draws,
        observed_case_counts=y,
        log_predictive_density=log_predictive_density,
        n_particles=P,
        final_particle_cloud=cloud if return_final_cloud else None,
    )