import math
from typing import Any, Optional
import numpy as np

from src.data import contact_matrix
from src.data import historical_population
from src.utils import load_config

config = load_config('config.yaml')


def age_shares(cfg: Any = config) -> np.ndarray:
    '''calculates age chares rebined to model age groups

    Parameters
    ----------
    cfg : Any, default=config
        configuration module or object.

    Returns
    -------
    np.ndarray
        2020 NHGIS age shares, rebinned to the model's age groups.
    '''
    age_population = contact_matrix.load_nhgis_age_population(
        cfg['data']['raw_nhgis_data_path'], cfg['data']['raw_nhgis_codebook_path']
    )
    pop_by_group = contact_matrix.population_by_age_group(age_population)
    return pop_by_group / pop_by_group.sum()


def _compartment_shares_for_group(maternal_immune_fraction: float = 0.05, recovered_fraction: Optional[float] = None, cfg: Any = config) -> tuple[float, float]:
    '''computes M and R fraction applied uniformly across age

    Parameters
    ----------
    maternal_immune_fraction : float, default=0.05
        fraction of newborns still maternally protected.
    recovered_fraction : Optional[float], default=None
        fraction of recovered individuals.
    cfg : Any, default=config
        configuration module or object.

    Returns
    -------
    tuple[float, float]
        tuple containing (maternal_immune_fraction, recovered_fraction).
    '''
    if recovered_fraction is None:
        r0_geo_mean = math.sqrt(cfg['literature']['r0_literature_range'][0] * cfg['literature']['r0_literature_range'][1])
        susceptible_equilibrium_fraction = 1.0 / r0_geo_mean
        recovered_fraction = 1.0 - maternal_immune_fraction - susceptible_equilibrium_fraction
    return maternal_immune_fraction, recovered_fraction


def prior_mean_initial_state(fit_window_start_year: float, seed_fraction: float, cfg: Any = config) -> dict[str, np.ndarray]:
    '''constructs deterministic low dimensional initial state before sampling noise

    Parameters
    ----------
    fit_window_start_year : float
        calendar year the fit window begins (e.g. 1920.0).
    seed_fraction : float
        fraction of the total population initially infectious (I0).
    cfg : Any, default=config
        configuration module or object.

    Returns
    -------
    dict[str, np.ndarray]
        dictionary with keys 'M0', 'S0', 'E0', 'I0', 'R0', each length A, in head-counts.
    '''
    A = cfg['age_structure']['n_age_groups']
    total_population = historical_population.population_for_year(int(round(fit_window_start_year)))
    shares = age_shares(cfg)
    N_a = shares * total_population

    m_frac, r_frac = _compartment_shares_for_group()

    I0 = seed_fraction * N_a
    E0 = seed_fraction * N_a
    remaining = np.clip(N_a - I0 - E0, 0.0, None)
    M0 = m_frac * remaining
    R0 = r_frac * remaining
    S0 = np.clip(remaining - M0 - R0, 0.0, None)

    return {'M0': M0, 'S0': S0, 'E0': E0, 'I0': I0, 'R0': R0}


def sample_initial_states(fit_window_start_year: float, seed_fraction: float, n_particles: int, rng: np.random.Generator, dirichlet_concentration: float = 200.0, cfg: Any = config) -> list[dict[str, np.ndarray]]:
    '''draws independent inital states. perturbs deterministic mean state with dirichlet noise and samples independent
    poisson draws for initial infectious pool

    Parameters
    ----------
    fit_window_start_year : float
        calendar year the fit window begins.
    seed_fraction : float
        fraction of the total population initially infectious.
    n_particles : int
        number of independent particles to draw.
    rng : np.random.Generator
        random number generator.
    dirichlet_concentration : float, default=200.0
        concentration parameter for Dirichlet distribution.
    cfg : Any, default=config
        configuration module or object.

    Returns
    -------
    list[dict[str, np.ndarray]]
        list of dicts with keys 'M0', 'S0', 'E0', 'I0', 'R0'.
    '''
    mean_ic = prior_mean_initial_state(fit_window_start_year, seed_fraction, cfg)
    A = cfg['age_structure']['n_age_groups']
    shares = age_shares(cfg)
    total_population = historical_population.population_for_year(int(round(fit_window_start_year)))
    m_frac, r_frac = _compartment_shares_for_group()

    draws = []
    for _ in range(n_particles):
        noisy_shares = rng.dirichlet(shares * dirichlet_concentration)
        N_a = noisy_shares * total_population

        mean_I0 = seed_fraction * N_a
        I0 = rng.poisson(np.clip(mean_I0, 1e-9, None)).astype(float)
        mean_E0 = seed_fraction * N_a
        E0 = rng.poisson(np.clip(mean_E0, 1e-9, None)).astype(float)

        remaining = np.clip(N_a - I0 - E0, 0.0, None)
        M0 = m_frac * remaining
        R0 = r_frac * remaining
        S0 = np.clip(remaining - M0 - R0, 0.0, None)

        draws.append({'M0': M0, 'S0': S0, 'E0': E0, 'I0': I0, 'R0': R0})
    return draws