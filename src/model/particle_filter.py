from dataclasses import dataclass
import os
from typing import Any, Callable, Optional, Union, cast
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.model import initial_state_prior, observation_model, stochastic_model
from src.utils import load_config

config = load_config('config.yaml')


@dataclass
class ParticleFilterResult:
    '''results from boostrap particle filter

    Attributes
    ----------
    log_likelihood : float
        estimated log-likelihood.
    ess_history : np.ndarray
        effective sample size at each observation time.
    resampled_history : np.ndarray
        boolean array indicating whether resampling occurred.
    final_states : list[stochastic_model.StochasticState]
        final resampled particle states.
    incidence_mean : np.ndarray
        particle-mean simulated incidence per interval.
    score_start_index : int, default=0
        first observation index counted toward log_likelihood.
    '''

    log_likelihood: float
    ess_history: np.ndarray # shape (T,) effective sample size at each obs time
    resampled_history: np.ndarray # shape (T,) bool, whether resampling fired
    final_states: list[stochastic_model.StochasticState]  # list[StochasticState], final particle cloud
    incidence_mean: np.ndarray # shape (T-1, A) particle-mean simulated incidence per interval
    score_start_index: int = 0 # first index (into observed_times) counted toward log_likelihood


def _resolve_score_start_index(observed_times: np.ndarray, score_start: Optional[float]) -> int:
    '''maps the year to start scoring into an index for the stochastic model to enable spin up
    Parameters
    ----------
    observed_times : np.ndarray
        array of observation times.
    score_start : Optional[float]
        elapsed-years cutoff for likelihood scoring.

    Returns
    -------
    int
        index into observed_times where scoring starts.
    '''
    if score_start is None:
        return 0
    idx = int(np.searchsorted(observed_times, score_start, side='left'))
    return min(max(idx, 0), len(observed_times))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''systematic resample of the particle cloud weights

    Parameters
    ----------
    weights : np.ndarray
        normalized particle weights array.
    rng : np.random.Generator
        random number generator.

    Returns
    -------
    np.ndarray
        array of resampled particle indices.
    '''
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against fp drift
    return np.searchsorted(cumulative, positions)


def run_particle_filter(theta: dict[str, float], observed_times: np.ndarray, observed_case_counts: np.ndarray, fixed_parameters: dict[str, Any], n_particles: int, rng: np.random.Generator, fit_window_start_year: float, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float), ess_resample_fraction: float = config['calibration']['particle_filter_ess_resample_fraction'], score_start: Optional[float] = None, return_diagnostics: bool = False, initial_cloud: Optional[stochastic_model.StochasticParticleCloud] = None) -> ParticleFilterResult:
    '''runs a bootstrap particle filter returning an unbiased estimate of log P_theta(Y_1:T) by propagating particles through the
    stochastic model. weights particles using negative-binomial observation model density.

    Parameters
    ----------
    theta : dict[str, float]
        free parameter estimates on natural scale.
    observed_times : np.ndarray
        elapsed years since fit window start of shape (T,).
    observed_case_counts : np.ndarray
        aggregate case counts array of shape (T,).
    fixed_parameters : dict[str, Any]
        fixed epidemiological parameters dictionary.
    n_particles : int
        number of particles in ensemble.
    rng : np.random.Generator
        random number generator.
    fit_window_start_year : float
        calendar year corresponding to start of observed times.
    dt : Optional[float], default=None
        integration sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        per-age-group importation rate.
    ess_resample_fraction : float, default=config['calibration']['particle_filter_ess_resample_fraction']
        ESS fraction threshold triggering resampling.
    score_start : Optional[float], default=None
        elapsed-years cutoff before which observations are not scored.
    return_diagnostics : bool, default=False
        whether to attach additional diagnostics.
    initial_cloud : Optional[stochastic_model.StochasticParticleCloud], default=None
        pre-existing particle cloud for warm start.

    Returns
    -------
    ParticleFilterResult
        bootstrap particle filter result object.

    Raises
    ------
    ValueError
        if `observed_times` and `observed_case_counts` have unequal lengths.
    '''
    A = config['age_structure']['n_age_groups']
    T = len(observed_times)
    if len(observed_case_counts) != T:
        raise ValueError('observed_times and observed_case_counts must be the same length')
    score_start_idx = _resolve_score_start_index(observed_times, score_start)

    birth_rate = _wrap_calendar_rate(fixed_parameters['birth_rate'], fit_window_start_year)
    death_rate = _wrap_calendar_rate(fixed_parameters['death_rate'], fit_window_start_year)

    parameters = {
        'beta_0': theta['beta_0'],
        'beta_1': theta['beta_1'],
        'phi': theta['phi'],
        'gamma': fixed_parameters['gamma'],
        'sigma': fixed_parameters['sigma'],
        'maternal_waning_rate': fixed_parameters['maternal_waning_rate'],
        'birth_rate': birth_rate,
        'death_rate': death_rate,
        'vaccination_rates': _wrap_calendar_rate(fixed_parameters['vaccination_rates'], fit_window_start_year),
        'contact_matrix': fixed_parameters['contact_matrix'],
    }
    rho = theta['rho']
    phi_obs = theta['phi_obs']

    if initial_cloud is not None:
        cloud = stochastic_model.StochasticParticleCloud(
            t=observed_times[0], counts=initial_cloud.counts.copy()
        )
        n_particles = cloud.n_particles
    else:
        ic_draws = initial_state_prior.sample_initial_states(
            fit_window_start_year=fit_window_start_year,
            seed_fraction=theta['seed_fraction'],
            n_particles=n_particles,
            rng=rng,
        )
        particle_states = [
            stochastic_model.make_initial_state(
                ic['M0'], ic['S0'], ic['E0'], ic['I0'], ic['R0'], t0=observed_times[0]
            )
            for ic in ic_draws
        ]
        cloud = stochastic_model.StochasticParticleCloud.from_states(particle_states)

    log_likelihood = 0.0
    ess_history = np.zeros(T)
    resampled_history = np.zeros(T, dtype=bool)
    incidence_mean = np.zeros((T - 1, A))
    ess_history[0] = float(n_particles)

    for k in range(1, T):
        duration = float(observed_times[k] - observed_times[k - 1])
        cloud, incidences = stochastic_model.simulate_interval_batch(
            cloud, duration, parameters, rng, dt=dt, importation_rate=importation_rate
        )

        incidence_mean[k - 1] = incidences.mean(axis=0)
        total_incidence = incidences.sum(axis=1)
        mu = rho * total_incidence

        log_weights = observation_model.negbin_logpmf(
            np.full(n_particles, observed_case_counts[k]), mu, phi_obs
        )

        max_lw = np.max(log_weights)
        if not np.isfinite(max_lw):
            log_likelihood = -np.inf
            ess_history[k:] = 0.0
            break

        shifted = np.exp(log_weights - max_lw)
        mean_weight = shifted.mean()
        increment = max_lw + np.log(mean_weight) if mean_weight > 0 else -np.inf
        if k >= score_start_idx:
            log_likelihood += increment
        elif not np.isfinite(increment):
            log_likelihood = -np.inf
        if not np.isfinite(log_likelihood):
            ess_history[k:] = 0.0
            break

        norm_weights = shifted / shifted.sum()
        ess = 1.0 / np.sum(norm_weights ** 2)
        ess_history[k] = ess

        if ess < ess_resample_fraction * n_particles:
            idx = _systematic_resample(norm_weights, rng)
            cloud = stochastic_model.StochasticParticleCloud(t=cloud.t, counts=cloud.counts[idx].copy())
            resampled_history[k] = True

    return ParticleFilterResult(
        log_likelihood=float(log_likelihood),
        ess_history=ess_history,
        resampled_history=resampled_history,
        final_states=cloud.to_states(),
        incidence_mean=incidence_mean,
        score_start_index=score_start_idx,
    )


def _systematic_resample_grouped(norm_weights: np.ndarray, needs_resample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''performs systematic resampling independently within each parameter group

    Parameters
    ----------
    norm_weights : np.ndarray
        within-group normalized weights array of shape (n_theta, n_inner).
    needs_resample : np.ndarray
        boolean array of shape (n_theta,) indicating which groups resample.
    rng : np.random.Generator
        random number generator.

    Returns
    -------
    np.ndarray
        resampled local indices array of shape (n_theta, n_inner).
    '''
    n_theta, n_inner = norm_weights.shape
    local_idx = np.tile(np.arange(n_inner), (n_theta, 1))
    for g in range(n_theta):
        if not needs_resample[g]:
            continue
        w = norm_weights[g]
        positions = (rng.random() + np.arange(n_inner)) / n_inner
        cumulative = np.cumsum(w)
        cumulative[-1] = 1.0  # guard against fp drift
        local_idx[g] = np.searchsorted(cumulative, positions)
    return local_idx


@dataclass
class BatchParticleFilterResult:
    '''store per theta results from particle filter batch runs

    Parameters
    ----------
    log_likelihood : np.ndarray
        log likelihood array of shape (n_theta,).
    ess_history : np.ndarray
        effective sample size history of shape (n_theta, T).
    resampled_history : np.ndarray
        boolean resampling history of shape (n_theta, T).
    score_start_index : int, default=0
        first index counted toward log_likelihood.
    '''
    log_likelihood: np.ndarray  # shape (n_theta,)
    ess_history: np.ndarray  # shape (n_theta, T)
    resampled_history: np.ndarray  # shape (n_theta, T), bool
    score_start_index: int = 0  # first index (into observed_times) counted toward log_likelihood


def run_particle_filter_batch(thetas: list[dict[str, float]], observed_times: np.ndarray, observed_case_counts: np.ndarray, fixed_parameters: dict[str, Any], n_inner_particles: int, rng: np.random.Generator, fit_window_start_year: float, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float), ess_resample_fraction: float = config['calibration']['particle_filter_ess_resample_fraction'], score_start: Optional[float] = None, progress_desc: Optional[str] = None) -> BatchParticleFilterResult:
    '''runs independent particle filters across multiple thetas using a fused simulation and propagating particles as a
    single cloud vector. retains independent likelihood increments and resampling per theta

    Parameters
    ----------
    thetas : list[dict[str, float]]
        list of parameter dictionaries.
    observed_times : np.ndarray
        observation times array of shape (T,).
    observed_case_counts : np.ndarray
        observed case counts array of shape (T,).
    fixed_parameters : dict[str, Any]
        fixed model parameters dictionary.
    n_inner_particles : int
        particles per theta parameter set.
    rng : np.random.Generator
        random number generator.
    fit_window_start_year : float
        fit window start calendar year.
    dt : Optional[float], default=None
        sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate per age group.
    ess_resample_fraction : float, default=config['calibration']['particle_filter_ess_resample_fraction']
        ESS fraction threshold triggering resampling.
    score_start : Optional[float], default=None
        scoring cutoff in elapsed years.
    progress_desc : Optional[str], default=None
        tqdm progress bar description string.

    Returns
    -------
    BatchParticleFilterResult
        batched particle filter result object containing arrays across all theta values.

    Raises
    ------
    ValueError
        if `observed_times` and `observed_case_counts` have unequal lengths.
    '''
    n_theta = len(thetas)
    T = len(observed_times)
    if len(observed_case_counts) != T:
        raise ValueError('observed_times and observed_case_counts must be the same length')
    score_start_idx = _resolve_score_start_index(observed_times, score_start)

    birth_rate = _wrap_calendar_rate(fixed_parameters['birth_rate'], fit_window_start_year)
    death_rate = _wrap_calendar_rate(fixed_parameters['death_rate'], fit_window_start_year)

    beta_0 = np.repeat([th['beta_0'] for th in thetas], n_inner_particles)
    beta_1 = np.repeat([th['beta_1'] for th in thetas], n_inner_particles)
    phi = np.repeat([th['phi'] for th in thetas], n_inner_particles)
    rho = np.repeat([th['rho'] for th in thetas], n_inner_particles)
    phi_obs = np.repeat([th['phi_obs'] for th in thetas], n_inner_particles)

    parameters = {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'phi': phi,
        'gamma': fixed_parameters['gamma'],
        'sigma': fixed_parameters['sigma'],
        'maternal_waning_rate': fixed_parameters['maternal_waning_rate'],
        'birth_rate': birth_rate,
        'death_rate': death_rate,
        'vaccination_rates': _wrap_calendar_rate(fixed_parameters['vaccination_rates'], fit_window_start_year),
        'contact_matrix': fixed_parameters['contact_matrix'],
    }

    particle_states = []
    for th in thetas:
        ic_draws = initial_state_prior.sample_initial_states(
            fit_window_start_year=fit_window_start_year,
            seed_fraction=th['seed_fraction'],
            n_particles=n_inner_particles,
            rng=rng,
        )
        particle_states.extend(
            stochastic_model.make_initial_state(
                ic['M0'], ic['S0'], ic['E0'], ic['I0'], ic['R0'], t0=observed_times[0]
            )
            for ic in ic_draws
        )
    cloud = stochastic_model.StochasticParticleCloud.from_states(particle_states)

    log_likelihood = np.zeros(n_theta)
    active = np.ones(n_theta, dtype=bool)
    ess_history = np.zeros((n_theta, T))
    resampled_history = np.zeros((n_theta, T), dtype=bool)
    ess_history[:, 0] = float(n_inner_particles)

    for k in tqdm(range(1, T), desc=progress_desc, leave=False, disable=progress_desc is None):
        duration = float(observed_times[k] - observed_times[k - 1])
        cloud, incidences = stochastic_model.simulate_interval_batch(
            cloud, duration, parameters, rng, dt=dt, importation_rate=importation_rate
        )

        total_incidence = incidences.sum(axis=1)
        mu = rho * total_incidence
        log_weights = observation_model.negbin_logpmf(
            np.full(cloud.n_particles, observed_case_counts[k]), mu, phi_obs
        )
        lw = log_weights.reshape(n_theta, n_inner_particles)

        max_lw = np.max(lw, axis=1)
        group_finite = np.isfinite(max_lw)
        shifted = np.where(
            group_finite[:, None], np.exp(lw - np.where(group_finite, max_lw, 0.0)[:, None]), 0.0
        )
        mean_weight = shifted.mean(axis=1)

        with np.errstate(divide='ignore'):
            increment = np.where(
                group_finite & (mean_weight > 0), max_lw + np.log(mean_weight), -np.inf
            )

        newly_dead = active & ~np.isfinite(increment)
        if k >= score_start_idx:
            log_likelihood = np.where(active, log_likelihood + np.where(active, increment, 0.0), log_likelihood)
        log_likelihood = np.where(newly_dead, -np.inf, log_likelihood)
        active = active & np.isfinite(increment)
        ess_history[newly_dead, k:] = 0.0

        weight_sum = shifted.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            norm_weights = np.where(weight_sum[:, None] > 0, shifted / np.where(weight_sum[:, None] > 0, weight_sum[:, None], 1.0), 1.0 / n_inner_particles)
        ess = 1.0 / np.sum(norm_weights ** 2, axis=1)
        ess_history[active, k] = ess[active]

        needs_resample = active & (ess < ess_resample_fraction * n_inner_particles)
        if np.any(needs_resample):
            local_idx = _systematic_resample_grouped(norm_weights, needs_resample, rng)
            group_offsets = (np.arange(n_theta) * n_inner_particles)[:, None]
            global_idx = (local_idx + group_offsets).reshape(-1)
            cloud = stochastic_model.StochasticParticleCloud(t=cloud.t, counts=cloud.counts[global_idx].copy())
            resampled_history[needs_resample, k] = True

        if not np.any(active):
            break

    return BatchParticleFilterResult(
        log_likelihood=log_likelihood,
        ess_history=ess_history,
        resampled_history=resampled_history,
        score_start_index=score_start_idx,
    )


def _run_batch_chunk(thetas: list[dict[str, float]], observed_times: np.ndarray, observed_case_counts: np.ndarray, fixed_parameters: dict[str, Any], n_inner_particles: int, seed_seq: np.random.SeedSequence, fit_window_start_year: float, dt: Optional[float], importation_rate: np.ndarray, ess_resample_fraction: float, score_start: Optional[float]) -> BatchParticleFilterResult:
    '''worker entry point for parallel batch particle filtering
    Parameters
    ----------
    thetas : list[dict[str, float]]
        chunk of parameter dictionaries.
    observed_times : np.ndarray
        observation times.
    observed_case_counts : np.ndarray
        observed case counts.
    fixed_parameters : dict[str, Any]
        fixed model parameters.
    n_inner_particles : int
        particles per theta.
    seed_seq : np.random.SeedSequence
        seed sequence for random generator creation.
    fit_window_start_year : float
        fit window start year.
    dt : Optional[float]
        sub-step size in years.
    importation_rate : np.ndarray
        importation rate array.
    ess_resample_fraction : float
        ESS fraction threshold for resampling.
    score_start : Optional[float]
        scoring cutoff time.

    Returns
    -------
    BatchParticleFilterResult
        chunk execution result object.
    '''
    rng = np.random.default_rng(seed_seq)
    return run_particle_filter_batch(
        thetas=thetas,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        n_inner_particles=n_inner_particles,
        rng=rng,
        fit_window_start_year=fit_window_start_year,
        dt=dt,
        importation_rate=importation_rate,
        ess_resample_fraction=ess_resample_fraction,
        score_start=score_start,
    )


def run_particle_filter_batch_parallel(thetas: list[dict[str, float]], observed_times: np.ndarray, observed_case_counts: np.ndarray, fixed_parameters: dict[str, Any], n_inner_particles: int, rng: np.random.Generator, fit_window_start_year: float, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float), ess_resample_fraction: float = config['calibration']['particle_filter_ess_resample_fraction'], score_start: Optional[float] = None, n_jobs: int = -1, verbose: int = 0) -> BatchParticleFilterResult:
    '''shards thetas into contiguous chunks and runs each batch filter in parallel
    Parameters
    ----------
    thetas : list[dict[str, float]]
        list of parameter dictionaries.
    observed_times : np.ndarray
        observation times array.
    observed_case_counts : np.ndarray
        observed case counts array.
    fixed_parameters : dict[str, Any]
        fixed model parameters dictionary.
    n_inner_particles : int
        particles per theta.
    rng : np.random.Generator
        random generator.
    fit_window_start_year : float
        fit window start year.
    dt : Optional[float], default=None
        sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate vector.
    ess_resample_fraction : float, default=config['calibration']['particle_filter_ess_resample_fraction']
        ESS fraction threshold for resampling.
    score_start : Optional[float], default=None
        scoring start cutoff time.
    n_jobs : int, default=-1
        number of parallel jobs.
    verbose : int, default=0
        joblib verbosity level.

    Returns
    -------
    BatchParticleFilterResult
        concatenated result object across all chunks.

    Raises
    ------
    ValueError
        if `n_jobs` resolves to less than one worker.
    '''
    n_theta = len(thetas)
    total_cores = os.cpu_count() or 1
    if n_jobs == -1:
        requested_workers = total_cores
    elif n_jobs < 0:
        requested_workers = max(1, total_cores + 1 + n_jobs)
    else:
        requested_workers = n_jobs
    if requested_workers < 1:
        raise ValueError('n_jobs must resolve to at least one worker')

    resolved_n_jobs = min(requested_workers, n_theta)
    n_chunks = resolved_n_jobs

    chunk_bounds = np.linspace(0, n_theta, n_chunks + 1, dtype=int)
    chunks = [
        thetas[chunk_bounds[c]:chunk_bounds[c + 1]]
        for c in range(n_chunks)
        if chunk_bounds[c + 1] > chunk_bounds[c]
    ]

    parent_seed_seq = np.random.SeedSequence(rng.integers(0, 2**63 - 1))
    child_seed_seqs = parent_seed_seq.spawn(len(chunks))

    chunk_results = cast(
        list[BatchParticleFilterResult],
        Parallel(n_jobs=resolved_n_jobs, backend='loky', verbose=verbose)(
            delayed(_run_batch_chunk)(
                chunk,
                observed_times,
                observed_case_counts,
                fixed_parameters,
                n_inner_particles,
                seed_seq,
                fit_window_start_year,
                dt,
                importation_rate,
                ess_resample_fraction,
                score_start,
            )
            for chunk, seed_seq in zip(chunks, child_seed_seqs)
        ),
    )

    return BatchParticleFilterResult(
        log_likelihood=np.concatenate([r.log_likelihood for r in chunk_results]),
        ess_history=np.concatenate([r.ess_history for r in chunk_results], axis=0),
        resampled_history=np.concatenate([r.resampled_history for r in chunk_results], axis=0),
        score_start_index=_resolve_score_start_index(observed_times, score_start),
    )


def _wrap_calendar_rate(rate: Union[float, Callable[[float], float]], fit_window_start_year: float) -> Union[float, Callable[[float], float]]:
    ''' wraps calendar year rate callable to accept elapsed simulation time

    Parameters
    ----------
    rate : Union[float, Callable[[float], float]]
        rate scalar or calendar-year callable.
    fit_window_start_year : float
        calendar start year.

    Returns
    -------
    Union[float, Callable[[float], float]]
        wrapped callable taking elapsed time, or unchanged scalar.
    '''
    if not callable(rate):
        return rate

    def wrapped(elapsed_t: float) -> float:
        return rate(fit_window_start_year + elapsed_t)

    return wrapped