from dataclasses import dataclass
from typing import Callable, Mapping, Optional
import numpy as np

from src.experiments import vaccination
from src.model import observation_model, particle_filter, stochastic_model
from src.utils import load_config

config = load_config('config.yaml')

BRANCH_YEAR = 2000.0
TODDLER_DOSE_CHANNEL = vaccination.TODDLER_DOSE_CHANNEL
SCHOOL_ENTRY_CHANNEL = vaccination.SCHOOL_ENTRY_CHANNEL


@dataclass
class CounterfactualResult:
    '''unconditioned posterior predictive output. no filtering or resampling against
    observed data so every particle is only forward simulation under scenario schedule

    Attributes
    ----------
    scenario : str
        name of the scenario run, one of the keys in EXPERIMENTS.
    calendar_years : np.ndarray
        calendar years at which the branch was evaluated.
    simulated_incidence : np.ndarray
        shape (n_particles, n_intervals, n_age_groups), per-interval
        incidence for every particle.
    simulated_case_draws : np.ndarray
        shape (n_particles, n_intervals), sampled observed case counts
        for every particle.
    vaccination_rates : np.ndarray
        shape (len(calendar_years), n_vaccination_rates), the scenario's
        vaccination-rate vector evaluated at every calendar year.
    final_particle_cloud : stochastic_model.StochasticParticleCloud
        state/parameter cloud at the end of the branch.
    '''
    scenario: str
    calendar_years: np.ndarray
    simulated_incidence: np.ndarray
    simulated_case_draws: np.ndarray
    vaccination_rates: np.ndarray
    final_particle_cloud: stochastic_model.StochasticParticleCloud

    @property
    def simulated_case_counts_mean(self) -> np.ndarray:
        '''averages simulated case draws across particle axis

        Returns
        -------
        np.ndarray, shape (n_intervals,)
            particle-mean simulated case count at each interval.
        '''
        return self.simulated_case_draws.mean(axis=0)

    @property
    def simulated_case_counts_ci(self) -> np.ndarray:
        '''uses particle clouds empirical spread as a posterior predictive interval rather than a parametric
        approximation. not an acutal credibility interval.

        Returns
        -------
        np.ndarray, shape (n_intervals, 2)
            5th/95th percentile simulated case count at each interval.
        '''
        return np.percentile(self.simulated_case_draws, [5, 95], axis=0).T


def _flat_schedule(toddler: float = 0.92, school_entry: float = 0.97) -> Callable[[float], np.ndarray]:
    '''continued high coverage scenario, used as a baseline

    Parameters
    ----------
    toddler : float
        constant toddler-dose coverage applied at every calendar year.
    school_entry : float
        constant school-entry coverage applied at every calendar year.

    Returns
    -------
    Callable[[float], np.ndarray]
        calendar-year -> vaccination-rate-vector function, constant in
        year.
    '''
    def schedule(calendar_year: float) -> np.ndarray:
        rates = np.zeros(config['age_structure']['n_vaccination_rates'], dtype=float)
        rates[TODDLER_DOSE_CHANNEL] = toddler
        rates[SCHOOL_ENTRY_CHANNEL] = school_entry
        return rates
    return schedule


def _delayed_schedule(toddler_coverage: float = 0.92, school_entry_coverage: float = 0.97, delayed_coverage: float = 0.2) -> Callable[[float], np.ndarray]:
    '''push vaccine coverage at reccomended age 1 back to toddler and school age

    Parameters
    ----------
    toddler_coverage : float
        total eventual toddler-dose coverage, split between the on-time
        and delayed sub-channels.
    school_entry_coverage : float
        constant school-entry coverage applied at every calendar year.
    delayed_coverage : float
        fraction of toddler_coverage routed to the delayed-dose
        channel (TODDLER_DOSE_CHANNEL + 1) rather than the on-time one.

    Returns
    -------
    Callable[[float], np.ndarray]
        calendar-year -> vaccination-rate-vector function, constant in
        year.
    '''
    def schedule(calendar_year: float) -> np.ndarray:
        rates = np.zeros(config['age_structure']['n_vaccination_rates'], dtype=float)
        rates[TODDLER_DOSE_CHANNEL] = delayed_coverage
        rates[TODDLER_DOSE_CHANNEL + 1] = toddler_coverage - delayed_coverage
        rates[SCHOOL_ENTRY_CHANNEL] = school_entry_coverage
        return rates
    return schedule


def _only_school_age(toddler_coverage: float = 0.2, school_entry_coverage: float = 0.97, delayed_coverage: float = 0.2) -> Callable[[float], np.ndarray]:
    '''only vaccinate at the school age

    Parameters
    ----------
    toddler_coverage : float
        total eventual toddler-dose coverage, split between the on-time
        and delayed sub-channels.
    school_entry_coverage : float
        constant school-entry coverage applied at every calendar year.
    delayed_coverage : float
        fraction of toddler_coverage routed to the delayed-dose
        channel (TODDLER_DOSE_CHANNEL + 1) rather than the on-time one.

    Returns
    -------
    Callable[[float], np.ndarray]
        calendar-year -> vaccination-rate-vector function, constant in
        year.
    '''
    def schedule(calendar_year: float) -> np.ndarray:
        rates = np.zeros(config['age_structure']['n_vaccination_rates'], dtype=float)
        rates[TODDLER_DOSE_CHANNEL] = delayed_coverage
        rates[TODDLER_DOSE_CHANNEL + 1] = toddler_coverage - delayed_coverage
        rates[SCHOOL_ENTRY_CHANNEL] = school_entry_coverage
        return rates
    return schedule


EXPERIMENTS: dict[str, Callable[[float], np.ndarray]] = {
    'continued_high_coverage': _flat_schedule(),
    'delayed_schedule': _delayed_schedule(),
    'only_school_age': _only_school_age(),
}


def _wrap_calendar_rates(parameters: dict, branch_year: float) -> dict:
    '''converts to elapsed simulation time for stochastic model. shallow copy rather than in
    place mutation

    Parameters
    ----------
    parameters : dict
        model parameters dict, must contain birth_rate, death_rate,
        and vaccination_rates.
    branch_year : float
        calendar year at which the branch's elapsed-time-0 begins.

    Returns
    -------
    dict
        copy of parameters with birth_rate, death_rate, and
        vaccination_rates wrapped to accept elapsed simulation time
        rather than calendar year.
    '''
    wrapped = dict(parameters)
    for key in ('birth_rate', 'death_rate', 'vaccination_rates'):
        wrapped[key] = particle_filter._wrap_calendar_rate(wrapped[key], branch_year)
    return wrapped


def run_branch(scenario: str, initial_cloud: stochastic_model.StochasticParticleCloud, theta_hat: Mapping[str, float], fixed_parameters: Mapping[str, object], calendar_years: np.ndarray, rng: np.random.Generator, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> CounterfactualResult:
    '''propagates a scenario forward with no filtering or resampling

    Parameters
    ----------
    scenario : str
        one of EXPERIMENTS' keys, selecting the vaccination schedule to
        run under.
    initial_cloud : stochastic_model.StochasticParticleCloud
        particle cloud to branch from, e.g. the terminal cloud from
        calibration.iterated_filtering or final_likelihood_evaluation.
    theta_hat : Mapping[str, float]
        calibrated parameter point (beta_0, beta_1, phi, rho,
        phi_obs), natural scale.
    fixed_parameters : Mapping[str, object]
        model parameters held fixed for calibration (spec section 9):
        gamma, sigma, maternal_waning_rate, birth_rate,
        death_rate, contact_matrix.
    calendar_years : np.ndarray
        calendar years at which to evaluate the branch, strictly
        increasing.
    rng : np.random.Generator
        random generator driving both the stochastic transitions and the
        observation-model case draws.
    dt : float, optional
        internal simulation step size, by default None (uses
        stochastic_model.simulate_interval_batch's own default).
    importation_rate : np.ndarray
        baseline annual importation hazard by age group, by default
        config['model']['default_importation_rate'].

    Returns
    -------
    CounterfactualResult
        the branch's simulated incidence, case draws, vaccination-rate
        grid, and terminal particle cloud.

    Raises
    ------
    ValueError
        if scenario is not a key of EXPERIMENTS.
    '''
    if scenario not in EXPERIMENTS:
        raise ValueError(f'Unknown scenario: {scenario}. Available: {list(EXPERIMENTS.keys())}')

    years = np.asarray(calendar_years, dtype=float)
    schedule = EXPERIMENTS[scenario]

    parameters = {
        'beta_0': theta_hat['beta_0'],
        'beta_1': theta_hat['beta_1'],
        'phi': theta_hat['phi'],
        'gamma': fixed_parameters['gamma'],
        'sigma': fixed_parameters['sigma'],
        'maternal_waning_rate': fixed_parameters['maternal_waning_rate'],
        'birth_rate': fixed_parameters['birth_rate'],
        'death_rate': fixed_parameters['death_rate'],
        'vaccination_rates': schedule,
        'contact_matrix': fixed_parameters['contact_matrix'],
    }
    parameters = _wrap_calendar_rates(parameters, BRANCH_YEAR)

    cloud = stochastic_model.StochasticParticleCloud(
        t=0.0, counts=initial_cloud.counts.copy()
    )
    P = cloud.n_particles
    A = config['age_structure']['n_age_groups']
    incidence = np.zeros((P, len(years) - 1, A), dtype=np.int64)

    for k in range(1, len(years)):
        duration = float(years[k] - years[k - 1])
        cloud, inc = stochastic_model.simulate_interval_batch(
            cloud, duration, parameters, rng, dt=dt, importation_rate=importation_rate
        )
        incidence[:, k - 1, :] = inc

    rho = float(theta_hat['rho'])
    phi_obs = float(theta_hat['phi_obs'])
    total_incidence = incidence.sum(axis=2)
    mu = rho * total_incidence
    case_draws = np.stack(
        [observation_model.negbin_rvs(mu[:, k], phi_obs, rng) for k in range(len(years) - 1)],
        axis=1,
    )

    vaccination_grid = np.stack([schedule(y) for y in years], axis=0)
    return CounterfactualResult(
        scenario=scenario,
        calendar_years=years,
        simulated_incidence=incidence,
        simulated_case_draws=case_draws,
        vaccination_rates=vaccination_grid,
        final_particle_cloud=cloud,
    )