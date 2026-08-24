from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.special import expit, logit
from tqdm import tqdm

from src.utils import load_config
from src.model import initial_state_prior, observation_model, particle_filter, stochastic_model

config = load_config('config.yaml')
# ['beta_0','beta_1','phi','rho','phi_obs','seed_fraction']
FREE_PARAMS = config['calibration']['stochastic_free_parameters']


def transform(theta: dict[str, float]) -> np.ndarray:
    '''transforms theta parameters into a scale for the optimizer. beta 0 and phi obs are log transformed
    since it must be positive and beta 1, rho, and seed fraction are logit transformed since
    they must be bound by [0, 1]

    Parameters
    ----------
    theta : dict[str, float]
        parameter values on the natural scale, keyed by name.

    Returns
    -------
    np.ndarray
        unconstrained vector on the IF2/optimizer scale, columns in
        FREE_PARAMS order.
    '''
    return np.array([
        np.log(theta['beta_0']),
        logit(np.clip(theta['beta_1'], 1e-9, 1 - 1e-9)),
        theta['phi'],
        logit(np.clip(theta['rho'], 1e-9, 1 - 1e-9)),
        np.log(theta['phi_obs']),
        logit(np.clip(theta['seed_fraction'], 1e-12, 1 - 1e-12)),
    ])


def untransform(x: np.ndarray) -> dict[str, float]:
    '''reverts transformation of parameters back to the natural scale. exponential for beta 0 and phi obs,
    mod 2pi for phi since it is the phase parameter

    Parameters
    ----------
    x : np.ndarray
        unconstrained vector on the IF2/optimizer scale, columns in
        FREE_PARAMS order.

    Returns
    -------
    dict[str, float]
        parameter values on the natural scale, keyed by name.
    '''
    beta_0 = np.exp(x[0])
    beta_1 = expit(x[1])
    phi = x[2] % (2 * np.pi)
    rho = expit(x[3])
    phi_obs = np.exp(x[4])
    seed_fraction = expit(x[5])
    return {
        'beta_0': float(beta_0),
        'beta_1': float(beta_1),
        'phi': float(phi),
        'rho': float(rho),
        'phi_obs': float(phi_obs),
        'seed_fraction': float(seed_fraction),
    }


def untransform_batch(x_batch: np.ndarray) -> np.ndarray:
    '''vectorized untransform for diagnostic logging

    Parameters
    ----------
    x_batch : np.ndarray
        shape (N, 6) array of unconstrained vectors, columns in
        FREE_PARAMS order.

    Returns
    -------
    np.ndarray
        shape (N, 6) array on the natural scale, columns in FREE_PARAMS
        order.
    '''
    beta_0 = np.exp(x_batch[:, 0])
    beta_1 = expit(x_batch[:, 1])
    phi = x_batch[:, 2] % (2 * np.pi)
    rho = expit(x_batch[:, 3])
    phi_obs = np.exp(x_batch[:, 4])
    seed_fraction = expit(x_batch[:, 5])
    return np.stack([beta_0, beta_1, phi, rho, phi_obs, seed_fraction], axis=1)


def sample_prior(rng: np.random.Generator, priors: dict = config['calibration']['priors']) -> dict[str, float]:
    '''prior families for parameters, defaults set in config
    Parameters
    ----------
    rng : np.random.Generator
        random generator used for all draws, for reproducibility.
    priors : dict, optional
        literature-informed prior specification per free parameter
        (spec sections 9-10), by default config['calibration']['priors'].

    Returns
    -------
    dict[str, float]
        one sampled theta on the natural scale, keyed by parameter name.
    '''
    out = {}
    for name in FREE_PARAMS:
        spec = priors[name]
        if spec['dist'] == 'lognormal':
            out[name] = float(np.exp(rng.normal(spec['mean_log'], spec['sd_log'])))
        elif spec['dist'] == 'logitnormal':
            out[name] = float(expit(rng.normal(spec['mean_logit'], spec['sd_logit'])))
        elif spec['dist'] == 'uniform':
            out[name] = float(rng.uniform(spec['low'], spec['high']))
        else:
            raise ValueError(f"unknown prior distribution {spec['dist']!r} for {name}")
    return out


def log_prior_density(theta: dict[str, float], priors: dict = config['calibration']['priors']) -> float:
    '''regularize the multistart and IF2 opbjectives as approximately maximum posterior searches around
    literature informed ranges rather than MLE that could wander on noisy evidence. lognormal and logitnormal
    terms include the jacobian of the transform so that density is correctly normalized. independence across
    parameters means that joint log denisty is a simple sum of parameter log densities

    Parameters
    ----------
    theta : dict[str, float]
        parameter values on the natural scale, keyed by name.
    priors : dict, optional
        literature-informed prior specification per free parameter
        (spec sections 9-10), by default config['calibration']['priors'].

    Returns
    -------
    float
        log p(theta) under the independent priors, natural scale. `-inf`
        if any parameter falls outside its prior's support.
    '''
    total = 0.0
    for name in FREE_PARAMS:
        spec = priors[name]
        v = theta[name]
        if spec['dist'] == 'lognormal':
            if v <= 0:
                return -np.inf
            z = (np.log(v) - spec['mean_log']) / spec['sd_log']
            total += -0.5 * z ** 2 - np.log(v * spec['sd_log'])
        elif spec['dist'] == 'logitnormal':
            if not (0 < v < 1):
                return -np.inf
            lv = logit(v)
            z = (lv - spec['mean_logit']) / spec['sd_logit']
            total += -0.5 * z ** 2 - np.log(v * (1 - v) * spec['sd_logit'])
        elif spec['dist'] == 'uniform':
            if not (spec['low'] <= v <= spec['high']):
                return -np.inf
            total += -np.log(spec['high'] - spec['low'])
    return total


def multistart_screen(
    observed_times: np.ndarray,
    observed_case_counts: np.ndarray,
    fixed_parameters: dict,
    fit_window_start_year: float,
    n_starts: int = config['calibration']['multistart_n_starts'],
    n_particles: int = config['calibration']['multistart_n_particles'],
    seed: int = config['calibration']['multistart_seed'],
    score_start: Optional[float] = None,
) -> list[tuple[dict[str, float], float]]:
    '''creates multistart using genuine priors rather than parameter bounds. all starts are
    simulated as one fused cloud to vectorize starts
    Parameters
    ----------
    observed_times : np.ndarray
        observation times.
    observed_case_counts : np.ndarray
        observed case counts at each observed time.
    fixed_parameters : dict
        model parameters held fixed for this calibration run (spec
        section 9): sigma, gamma, maternal_waning_rate, contact matrix,
        birth_rate/death_rate functions.
    fit_window_start_year : float
        calendar year corresponding to `observed_times[0]`, used to wrap
        calendar-indexed vital rate functions.
    n_starts : int, optional
        number of prior draws to screen, by default config['calibration']['multistart_n_starts'].
    n_particles : int, optional
        particle count used for the cheap per-start likelihood estimate,
        by default config['calibration']['multistart_n_particles'].
    seed : int, optional
        seed for the master RNG controlling prior draws and the run RNG,
        by default config['calibration']['multistart_seed'].
    score_start : float, optional
        calendar time after which the likelihood is scored, by default None
        (score from the first observation).

    Returns
    -------
    list[tuple[dict[str, float], float]]
        (theta, log_posterior) pairs, sorted best-first (highest
        log_posterior).
    '''
    master_rng = np.random.default_rng(seed)
    thetas = [sample_prior(master_rng) for _ in range(n_starts)]
    run_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    batch_result = particle_filter.run_particle_filter_batch_parallel(
        thetas=thetas,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        n_inner_particles=n_particles,
        rng=run_rng,
        fit_window_start_year=fit_window_start_year,
        score_start=score_start,
        verbose=5,
    )
    scored = [
        (theta, float(batch_result.log_likelihood[i]) + log_prior_density(theta))
        for i, theta in enumerate(thetas)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


@dataclass
class IF2Result:
    '''data class storing the results of IF2 fitting

    Attributes
    ----------
    theta_hat : dict[str, float]
        best-seen-iteration parameter estimate (by log-likelihood), not
        necessarily the last iteration.
    theta_hat_last_iteration : dict[str, float]
        filtering mean of the terminal parameter population from the
        last iteration, matching ``mif2``'s usual convention.
    best_iteration : int
        zero-based iteration with the highest finite perturbed-model log
        likelihood.
    theta_trace : np.ndarray
        shape (n_iterations, n_parameters), final filtering mean each
        iteration.
    log_likelihood_trace : np.ndarray
        shape (n_iterations,), perturbed-model particle-filter log
        likelihood.
    particle_theta_trace : np.ndarray
        shape (n_iterations, n_particles, n_parameters), pre-resampling
        terminal cloud, natural scale.
    particle_weight_trace : np.ndarray
        shape (n_iterations, n_particles), normalized terminal
        observation weights.
    ess_trace : np.ndarray
        shape (n_iterations, n_observation_intervals), ESS before each
        resampling step.
    resampled_trace : np.ndarray
        shape (n_iterations, n_observation_intervals), always true for
        successful updates.
    '''
    theta_hat: dict[str, float]
    theta_hat_last_iteration: dict[str, float]
    best_iteration: int
    theta_trace: np.ndarray
    log_likelihood_trace: np.ndarray
    particle_theta_trace: np.ndarray
    particle_weight_trace: np.ndarray
    ess_trace: np.ndarray
    resampled_trace: np.ndarray


def _if2_cooling_scale(iteration: int, interval_index: int, n_intervals: int, cooling_fraction: float) -> float:
    '''implements geometric cooling schedule (n-1 + (m-1)N)/(50N)

    Parameters
    ----------
    iteration : int
        zero-based IF2 iteration index.
    interval_index : int
        zero-based observation interval index within the iteration.
    n_intervals : int
        total number of observation intervals per iteration.
    cooling_fraction : float
        the ``cooling.fraction.50`` parameter: the geometric factor by
        which the perturbation SD shrinks over 50 full passes through
        the data.

    Returns
    -------
    float
        multiplicative cooling factor applied to the initial
        perturbation SD.
    '''
    if not (0.0 < cooling_fraction <= 1.0):
        raise ValueError('cooling_fraction must be in (0, 1]')
    return float(cooling_fraction ** ((interval_index + iteration * n_intervals) / (50.0 * n_intervals)))


def iterated_filtering(
    theta0: dict[str, float],
    observed_times: np.ndarray,
    observed_case_counts: np.ndarray,
    fixed_parameters: dict,
    fit_window_start_year: float,
    n_iterations: int = config['calibration']['if2_n_iterations'],
    n_particles: int = config['calibration']['if2_n_particles'],
    initial_perturbation_sd: dict = config['calibration']['if2_initial_perturbation_sd'],
    cooling_fraction: float = config['calibration']['if2_cooling_fraction'],
    seed: int = 0,
    score_start: Optional[float] = None,
) -> IF2Result:
    '''implements IF2 iterated filtering as used in documented pomp models. each particle carries its
    own parameter vector and state. parameters are perturbed immediately before every observation interval
    while the inital value parameter is perturned only before the initial state draw.

    Parameters
    ----------
    theta0 : dict[str, float]
        starting parameter values on the natural scale.
    observed_times : np.ndarray
        observation times, strictly increasing, length >= 2.
    observed_case_counts : np.ndarray
        observed case counts, same length as `observed_times`.
    fixed_parameters : dict
        model parameters held fixed for this calibration run (spec
        section 9): sigma, gamma, maternal_waning_rate, contact matrix,
        birth_rate/death_rate functions, vaccination_rates.
    fit_window_start_year : float
        calendar year corresponding to `observed_times[0]`, used to wrap
        calendar-indexed vital rate functions.
    n_iterations : int, optional
        number of IF2 iterations, by default config['calibration']['if2_n_iterations'].
    n_particles : int, optional
        particle count per iteration, by default config['calibration']['if2_n_particles'].
    initial_perturbation_sd : dict, optional
        initial random-walk perturbation SD per transformed parameter,
        keyed by name in `rw_names`, by default
        config['calibration']['if2_initial_perturbation_sd'].
    cooling_fraction : float, optional
        the ``cooling.fraction.50`` cooling schedule parameter, by
        default config['calibration']['if2_cooling_fraction'].
    seed : int, optional
        seed for the RNG driving perturbation and resampling, by default 0.
    score_start : float, optional
        calendar time after which the likelihood is scored, by default None
        (score from the first observation).

    Returns
    -------
    IF2Result
        full IF2 trace, best/last theta estimates, and diagnostics.

    Raises
    ------
    valueError
        if `observed_times` has fewer than two entries, if
        `observed_times` and `observed_case_counts` differ in length, if
        `n_particles` or `n_iterations` are too small, if
        `cooling_fraction` is outside (0, 1], if `score_start` resolves
        to an invalid index, or if any perturbation SD is negative or
        non-finite.
    keyError
        if `initial_perturbation_sd` is missing any required
        transformed-parameter name.
    runtimeError
        if the particle cloud goes extinct mid-iteration, or if no
        iteration produces a finite likelihood.
    '''
    rng = np.random.default_rng(seed)
    x_initial = transform(theta0)
    n_theta = len(x_initial)
    T = len(observed_times)
    n_intervals = T - 1
    if T < 2:
        raise ValueError('IF2 requires at least two observation times')
    if len(observed_case_counts) != T:
        raise ValueError('observed_times and observed_case_counts must have the same length')
    if n_particles < 2:
        raise ValueError('IF2 requires at least two particles')
    if n_iterations < 1:
        raise ValueError('IF2 requires at least one iteration')
    if not (0.0 < cooling_fraction <= 1.0):
        raise ValueError('cooling_fraction must be in (0, 1]')

    rw_names = [
        'log_beta_0', 'logit_beta_1', 'phi', 'logit_rho',
        'log_phi_obs', 'logit_seed_fraction',
    ]
    required = set(rw_names)
    missing = required.difference(initial_perturbation_sd)
    if missing:
        raise KeyError(f'missing IF2 perturbation SDs: {sorted(missing)}')
    sd0 = np.array([initial_perturbation_sd[name] for name in rw_names], dtype=float)
    if np.any(sd0 < 0) or not np.isfinite(sd0).all():
        raise ValueError('IF2 perturbation SDs must be finite and non-negative')

    score_start_idx = particle_filter._resolve_score_start_index(observed_times, score_start)
    if score_start_idx < 0 or score_start_idx > T:
        raise ValueError('invalid score_start index')

    theta_trace = np.zeros((n_iterations, n_theta))
    loglik_trace = np.full(n_iterations, -np.inf)
    particle_theta_trace = np.zeros((n_iterations, n_particles, n_theta))
    particle_weight_trace = np.full((n_iterations, n_particles), 1.0 / n_particles)
    ess_trace = np.full((n_iterations, n_intervals), np.nan)
    resampled_trace = np.zeros((n_iterations, n_intervals), dtype=bool)

    # starts with an identical parameter population and then carries forward identical particles into the next iteration
    x_particles = np.repeat(x_initial[None, :], n_particles, axis=0)

    for m in tqdm(range(n_iterations), desc='Iterated filtering', leave=False):
        loglik = 0.0
        alive = True
        failure_interval = None
        weights = np.full(n_particles, 1.0 / n_particles)

        cooling0 = _if2_cooling_scale(m, 0, n_intervals, cooling_fraction)
        step0 = rng.normal(0.0, sd0 * cooling0, size=(n_particles, n_theta))
        ivp_step = np.zeros_like(step0)
        ivp_step[:, 5] = step0[:, 5]
        x_particles = x_particles + ivp_step

        thetas = [untransform(x) for x in x_particles]
        states = []
        for th in thetas:
            ic = initial_state_prior.sample_initial_states(
                fit_window_start_year=fit_window_start_year,
                seed_fraction=th['seed_fraction'],
                n_particles=1,
                rng=rng,
            )[0]
            states.append(stochastic_model.make_initial_state(
                ic['M0'], ic['S0'], ic['E0'], ic['I0'], ic['R0'],
                t0=observed_times[0],
            ))
        cloud = stochastic_model.StochasticParticleCloud.from_states(states)

        terminal_x_pre_resample = None
        terminal_w_pre_resample = None

        for k in range(n_intervals):
            # k is zero-based observation interval
            cooling = _if2_cooling_scale(m, k, n_intervals, cooling_fraction)
            step = rng.normal(0.0, sd0 * cooling, size=(n_particles, n_theta))
            step[:, 5] = 0.0
            x_particles = x_particles + step
            thetas = [untransform(x) for x in x_particles]

            parameters = {
                'beta_0': np.array([th['beta_0'] for th in thetas]),
                'beta_1': np.array([th['beta_1'] for th in thetas]),
                'phi': np.array([th['phi'] for th in thetas]),
                'gamma': fixed_parameters['gamma'],
                'sigma': fixed_parameters['sigma'],
                'maternal_waning_rate': fixed_parameters['maternal_waning_rate'],
                'birth_rate': particle_filter._wrap_calendar_rate(
                    fixed_parameters['birth_rate'], fit_window_start_year
                ),
                'death_rate': particle_filter._wrap_calendar_rate(
                    fixed_parameters['death_rate'], fit_window_start_year
                ),
                'vaccination_rates': fixed_parameters['vaccination_rates'],
                'contact_matrix': fixed_parameters['contact_matrix'],
            }

            duration = float(observed_times[k + 1] - observed_times[k])
            if duration <= 0:
                raise ValueError('observed_times must be strictly increasing')
            cloud, incidences = stochastic_model.simulate_interval_batch(
                cloud, duration, parameters, rng
            )

            total_incidence = incidences.sum(axis=1)
            rho = np.array([th['rho'] for th in thetas])
            phi_obs = np.array([th['phi_obs'] for th in thetas])
            log_obs = observation_model.negbin_logpmf(
                np.full(n_particles, observed_case_counts[k + 1]),
                rho * total_incidence,
                phi_obs,
            )

            # bootstrap article filter likelihood increment
            max_log_obs = np.max(log_obs)
            if not np.isfinite(max_log_obs):
                alive = False
                failure_interval = k + 1
                ess_trace[m, k] = 0.0
                break

            scaled = np.exp(log_obs - max_log_obs)
            mean_scaled = scaled.mean()
            if not np.isfinite(mean_scaled) or mean_scaled <= 0.0:
                alive = False
                failure_interval = k + 1
                ess_trace[m, k] = 0.0
                break

            increment = max_log_obs + np.log(mean_scaled)
            if k + 1 >= score_start_idx:
                loglik += increment

            norm = scaled / scaled.sum()
            ess = 1.0 / np.sum(norm ** 2)
            ess_trace[m, k] = ess

            if k == n_intervals - 1:
                terminal_x_pre_resample = x_particles.copy()
                terminal_w_pre_resample = norm.copy()

            idx = particle_filter._systematic_resample(norm, rng)
            cloud = stochastic_model.StochasticParticleCloud(
                t=cloud.t,
                counts=cloud.counts[idx].copy(),
            )
            x_particles = x_particles[idx].copy()
            weights = norm.copy()
            resampled_trace[m, k] = True

        if alive and terminal_x_pre_resample is not None and terminal_w_pre_resample is not None:
            terminal_w_pre_resample = terminal_w_pre_resample / terminal_w_pre_resample.sum()
            particle_theta_trace[m] = untransform_batch(terminal_x_pre_resample)
            particle_weight_trace[m] = terminal_w_pre_resample
            theta_trace[m] = np.sum(
                terminal_w_pre_resample[:, None] * terminal_x_pre_resample, axis=0
            )
            loglik_trace[m] = loglik
        else:
            # failed iteration is a hard filtering failure
            raise RuntimeError(
                f'IF2 particle cloud became incompatible with the data at '
                f'iteration {m + 1}, observation interval {failure_interval}.'
            )

    theta_hat_last = untransform(theta_trace[-1])
    finite_iters = np.flatnonzero(np.isfinite(loglik_trace))
    if finite_iters.size == 0:
        raise RuntimeError('IF2 produced no finite likelihood-bearing iteration')
    best_iteration = int(finite_iters[np.argmax(loglik_trace[finite_iters])])
    theta_hat = untransform(theta_trace[best_iteration])

    return IF2Result(
        theta_hat=theta_hat,
        theta_hat_last_iteration=theta_hat_last,
        best_iteration=best_iteration,
        theta_trace=theta_trace,
        log_likelihood_trace=loglik_trace,
        particle_theta_trace=particle_theta_trace,
        particle_weight_trace=particle_weight_trace,
        ess_trace=ess_trace,
        resampled_trace=resampled_trace,
    )


def final_likelihood_evaluation(
    theta_hat: dict[str, float],
    observed_times: np.ndarray,
    observed_case_counts: np.ndarray,
    fixed_parameters: dict,
    fit_window_start_year: float,
    n_particles: int = config['calibration']['particle_filter_n_particles_final'],
    n_replicates: int = 5,
    seed: int = 1,
    score_start: Optional[float] = None,
) -> dict:
    '''repreats final likelihood evaluation multiple times since a single particle filter call is a noisy
    monte carlo estimator
    Parameters
    ----------
    theta_hat : dict[str, float]
        parameter point to evaluate, natural scale.
    observed_times : np.ndarray
        observation times.
    observed_case_counts : np.ndarray
        observed case counts at each observed time.
    fixed_parameters : dict
        model parameters held fixed for this calibration run (spec
        section 9).
    fit_window_start_year : float
        calendar year corresponding to `observed_times[0]`, used to wrap
        calendar-indexed vital rate functions.
    n_particles : int, optional
        particle count per replicate, by default
        config['calibration']['particle_filter_n_particles_final'].
    n_replicates : int, optional
        number of independent particle-filter replicates, by default 5.
    seed : int, optional
        seed for the RNG controlling the per-replicate run RNG, by default 1.
    score_start : float, optional
        calendar time after which the likelihood is scored, by default None
        (score from the first observation).

    Returns
    -------
    dict
        Keys: log_likelihood_replicates, log_likelihood_mean,
        log_likelihood_se, n_particles, n_replicates, n_extinct.
    '''
    rng = np.random.default_rng(seed)
    run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    batch_result = particle_filter.run_particle_filter_batch_parallel(
        thetas=[theta_hat] * n_replicates,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        n_inner_particles=n_particles,
        rng=run_rng,
        fit_window_start_year=fit_window_start_year,
        score_start=score_start,
        verbose=5,
    )
    logliks = batch_result.log_likelihood
    finite = logliks[np.isfinite(logliks)]
    return {
        'log_likelihood_replicates': logliks.tolist(),
        'log_likelihood_mean': float(finite.mean()) if len(finite) else float('-inf'),
        'log_likelihood_se': float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else None,
        'n_particles': n_particles,
        'n_replicates': n_replicates,
        'n_extinct': int(n_replicates - len(finite)),
    }