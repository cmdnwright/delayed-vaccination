from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
import numpy as np

from src.utils import load_config

config = load_config('config.yaml')

RateLike = Union[float, Callable[[float], float]]
_M, _S, _E, _I, _R = 0, 1, 2, 3, 4


@dataclass
class StochasticState:
    '''integer MSEIR state

    Parameters
    ----------
    t : float
        elapsed simulation time in years.
    counts : np.ndarray
        integer array of shape (5, A) holding compartment headcounts.
    '''
    t: float
    counts: np.ndarray

    def copy(self) -> 'StochasticState':
        '''depp copy of the state vector
        Returns
        -------
        StochasticState
            a copy of the current state.
        '''
        return StochasticState(self.t, self.counts.copy())

    @property
    def M(self) -> np.ndarray:
        '''extract maternal immune

        Returns
        -------
        np.ndarray
            maternal immune compartment counts.
        '''
        return self.counts[_M]

    @property
    def S(self) -> np.ndarray:
        '''extracts susceptibles

        Returns
        -------
        np.ndarray
            susceptible compartment counts.
        '''
        return self.counts[_S]

    @property
    def E(self) -> np.ndarray:
        '''extracts exposed

        Returns
        -------
        np.ndarray
            exposed compartment counts.
        '''
        return self.counts[_E]

    @property
    def I(self) -> np.ndarray:
        '''extracts infecteds

        Returns
        -------
        np.ndarray
            infectious compartment counts.
        '''
        return self.counts[_I]

    @property
    def R(self) -> np.ndarray:
        '''extracts recovered

        Returns
        -------
        np.ndarray
            recovered compartment counts.
        '''
        return self.counts[_R]

    @property
    def N(self) -> np.ndarray:
        '''total population per age group

        Returns
        -------
        np.ndarray
            total population per age group.
        '''
        return self.counts.sum(axis=0)


def make_initial_state(M0: np.ndarray, S0: np.ndarray, E0: np.ndarray, I0: np.ndarray, R0: np.ndarray, t0: float = 0.0) -> StochasticState:
    '''builds a StochasticState from five per group arrays after rounding to positive integers

    Parameters
    ----------
    M0 : np.ndarray
        initial maternal immune headcounts.
    S0 : np.ndarray
        initial susceptible headcounts.
    E0 : np.ndarray
        initial exposed headcounts.
    I0 : np.ndarray
        initial infectious headcounts.
    R0 : np.ndarray
        initial recovered headcounts.
    t0 : float, default=0.0
        initial simulation time in years.

    Returns
    -------
    StochasticState
        constructed stochastic state.
    '''
    counts = np.stack(
        [np.round(np.asarray(x, dtype=float)) for x in (M0, S0, E0, I0, R0)]
    ).astype(np.int64)
    counts = np.clip(counts, 0, None)
    return StochasticState(t=t0, counts=counts)


def _resolve_rate(rate: RateLike, t: float) -> float:
    '''evaluates callable rates like births and deaths and vaccination at time t

    Parameters
    ----------
    rate : RateLike
        scalar or rate callable.
    t : float
        simulation time in years.

    Returns
    -------
    float
        evaluated rate value.
    '''
    return rate(t) if callable(rate) else rate


def euler_multinomial_step(state: StochasticState, dt: float, parameters: dict[str, Any], rng: np.random.Generator, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> tuple[StochasticState, np.ndarray]:
    '''propagates stochastic state by one step using the euler multinomial step algorithm

    Parameters
    ----------
    state : StochasticState
        current state.
    dt : float
        step size in years.
    parameters : dict[str, Any]
        model parameters dictionary.
    rng : np.random.Generator
        random number generator.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        per-age-group importation hazard rate.

    Returns
    -------
    tuple[StochasticState, np.ndarray]
        tuple containing updated state and incidence array.
    '''
    A = config['age_structure']['n_age_groups']
    M, S, E, I, R = state.M, state.S, state.E, state.I, state.R
    t = state.t

    beta_0 = parameters['beta_0']
    beta_1 = parameters['beta_1']
    phi = parameters['phi']
    gamma = parameters['gamma']
    sigma = parameters['sigma']
    delta = parameters['maternal_waning_rate']
    birth_rate = _resolve_rate(parameters['birth_rate'], t)
    death_rate = _resolve_rate(parameters['death_rate'], t)
    v = np.asarray(_resolve_rate(parameters['vaccination_rates'], t), dtype=float)
    C = np.asarray(parameters['contact_matrix'], dtype=float)

    lam = force_of_infection(
        M.astype(float), S.astype(float), E.astype(float), I.astype(float), R.astype(float),
        t, beta_0, beta_1, phi, C,
    )

    g = _aging_rates()

    new_M = np.zeros(A, dtype=np.int64)
    new_S = np.zeros(A, dtype=np.int64)
    new_E = np.zeros(A, dtype=np.int64)
    new_I = np.zeros(A, dtype=np.int64)
    new_R = np.zeros(A, dtype=np.int64)
    incidence = np.zeros(A, dtype=np.int64)

    def _multinomial_draw(count: int, rates: np.ndarray) -> np.ndarray:
        if count <= 0:
            return np.zeros(len(rates) + 1, dtype=np.int64)
        total_rate = rates.sum()
        if total_rate <= 0:
            out = np.zeros(len(rates) + 1, dtype=np.int64)
            out[-1] = count
            return out
        p_leave_total = -np.expm1(-total_rate * dt)
        p_leave_total = min(max(p_leave_total, 0.0), 1.0)
        p_channels = (rates / total_rate) * p_leave_total
        p_stay = max(1.0 - p_channels.sum(), 0.0)
        pvals = np.concatenate([p_channels, [p_stay]])
        pvals = pvals / pvals.sum()
        return rng.multinomial(count, pvals)

    for a in range(A):
        has_next = a < A - 1
        g_a = g[a]

        rates = [death_rate, delta] + ([g_a] if has_next else [])
        draw = _multinomial_draw(int(M[a]), np.array(rates, dtype=float))
        n_wane = draw[1]
        n_age_M = draw[2] if has_next else 0
        new_M[a] += draw[-1]
        new_S[a] += n_wane
        if has_next:
            new_M[a + 1] += n_age_M

        if has_next:
            rates = [death_rate, lam[a], g_a * (1.0 - v[a]), g_a * v[a]]
        else:
            rates = [death_rate, lam[a]]
        draw = _multinomial_draw(int(S[a]), np.array(rates, dtype=float))
        n_infect = draw[1]
        new_E[a] += n_infect
        if has_next:
            n_age_S_novacc = draw[2]
            n_age_S_vacc = draw[3]
            new_S[a + 1] += n_age_S_novacc
            new_R[a + 1] += n_age_S_vacc
        new_S[a] += draw[-1]

        rates = [death_rate, sigma] + ([g_a] if has_next else [])
        draw = _multinomial_draw(int(E[a]), np.array(rates, dtype=float))
        n_progress = draw[1]
        incidence[a] += n_progress
        new_I[a] += n_progress
        if has_next:
            new_E[a + 1] += draw[2]
        new_E[a] += draw[-1]

        rates = [death_rate, gamma] + ([g_a] if has_next else [])
        draw = _multinomial_draw(int(I[a]), np.array(rates, dtype=float))
        n_recover = draw[1]
        new_R[a] += n_recover
        if has_next:
            new_I[a + 1] += draw[2]
        new_I[a] += draw[-1]

        rates = [death_rate] + ([g_a] if has_next else [])
        draw = _multinomial_draw(int(R[a]), np.array(rates, dtype=float))
        if has_next:
            new_R[a + 1] += draw[1]
        new_R[a] += draw[-1]

    N_total = float((M + S + E + I + R).sum())
    mean_births = birth_rate * N_total * dt
    n_births = rng.poisson(mean_births) if mean_births > 0 else 0
    new_M[0] += n_births

    expected_importations = importation_rate * dt
    n_imported = rng.poisson(expected_importations)

    n_imported = np.minimum(n_imported, new_S)
    new_S -= n_imported
    new_E += n_imported
    incidence += n_imported

    new_counts = np.stack([new_M, new_S, new_E, new_I, new_R])
    return StochasticState(t=t + dt, counts=new_counts), incidence


def simulate_interval(state: StochasticState, duration: float, parameters: dict[str, Any], rng: np.random.Generator, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> tuple[StochasticState, np.ndarray]:
    '''propagates stochastic state across duration of years subsets

    Parameters
    ----------
    state : StochasticState
        starting state.
    duration : float
        years to advance simulation.
    parameters : dict[str, Any]
        model parameter dictionary.
    rng : np.random.Generator
        random number generator.
    dt : Optional[float], default=None
        sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate vector.

    Returns
    -------
    tuple[StochasticState, np.ndarray]
        tuple containing final state and total interval incidence array.
    '''
    if dt is None:
        dt = float(config['model']['stochastic_dt_years'])
    n_steps = max(1, int(np.ceil(duration / dt)))
    actual_dt = duration / n_steps

    total_incidence = np.zeros(config['age_structure']['n_age_groups'], dtype=np.int64)
    current = state
    for _ in range(n_steps):
        current, inc = euler_multinomial_step(
            current, actual_dt, parameters, rng, importation_rate=importation_rate
        )
        total_incidence += inc
    return current, total_incidence


def _multinomial_draw_batch(counts: np.ndarray, pvals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''vectorized per row multinoial draws across the particle cloud

    Parameters
    ----------
    counts : np.ndarray
        initial occupancy counts array of shape (P,).
    pvals : np.ndarray
        normalized probabilities array of shape (P, C).
    rng : np.random.Generator
        random number generator.

    Returns
    -------
    np.ndarray
        sampled departure counts array of shape (P, C).
    '''
    P, C = pvals.shape
    draws = np.zeros((P, C), dtype=np.int64)
    remaining = counts.astype(np.int64).copy()
    remaining_prob = np.ones(P, dtype=float)
    for j in range(C - 1):
        with np.errstate(invalid='ignore', divide='ignore'):
            p_j = np.where(remaining_prob > 0, pvals[:, j] / remaining_prob, 0.0)
        p_j = np.clip(p_j, 0.0, 1.0)
        p_j = np.where(remaining > 0, p_j, 0.0)
        n_j = rng.binomial(remaining, p_j)
        draws[:, j] = n_j
        remaining = remaining - n_j
        remaining_prob = remaining_prob - pvals[:, j]
        remaining_prob = np.clip(remaining_prob, 0.0, None)
    draws[:, -1] = remaining
    return draws


def _channel_pvals_batch(rates: np.ndarray, dt: float) -> np.ndarray:
    '''per particle competing risk probabilities

    Parameters
    ----------
    rates : np.ndarray
        competing hazard rates array of shape (P, k).
    dt : float
        sub-step size in years.

    Returns
    -------
    np.ndarray
        channel probabilities array of shape (P, k+1).
    '''
    total_rate = rates.sum(axis=1)
    p_leave_total = -np.expm1(-total_rate * dt)
    p_leave_total = np.clip(p_leave_total, 0.0, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        p_channels = np.where(
            total_rate[:, None] > 0,
            (rates / np.where(total_rate[:, None] > 0, total_rate[:, None], 1.0)) * p_leave_total[:, None],
            0.0,
        )
    p_stay = np.clip(1.0 - p_channels.sum(axis=1), 0.0, None)
    pvals = np.concatenate([p_channels, p_stay[:, None]], axis=1)
    pvals = pvals / pvals.sum(axis=1, keepdims=True)
    return pvals


@dataclass
class StochasticParticleCloud:
    '''batched integer MSEIR state for particle ensemble

    Parameters
    ----------
    t : float
        elapsed simulation time in years.
    counts : np.ndarray
        counts array of shape (P, 5, A).
    '''

    t: float
    counts: np.ndarray

    @property
    def n_particles(self) -> int:
        '''
        Returns
        -------
        int
            number of particles in cloud.
        '''
        return self.counts.shape[0]

    def to_states(self) -> list[StochasticState]:
        '''
        Returns
        -------
        list[StochasticState]
            list of individual particle state objects.
        '''
        return [StochasticState(self.t, self.counts[i]) for i in range(self.n_particles)]

    @classmethod
    def from_states(cls, states: list[StochasticState]) -> 'StochasticParticleCloud':
        '''
        Parameters
        ----------
        states : list[StochasticState]
            list of particle states.

        Returns
        -------
        StochasticParticleCloud
            batched particle cloud object.
        '''
        t = states[0].t
        counts = np.stack([s.counts for s in states])
        return cls(t=t, counts=counts)


def euler_multinomial_step_batch(cloud: StochasticParticleCloud, dt: float, parameters: dict[str, Any], rng: np.random.Generator, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> tuple[StochasticParticleCloud, np.ndarray]:
    '''batch multinomial step algorithm

    Parameters
    ----------
    cloud : StochasticParticleCloud
        batched particle cloud.
    dt : float
        step size in years.
    parameters : dict[str, Any]
        model parameter dictionary.
    rng : np.random.Generator
        random generator.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate vector.

    Returns
    -------
    tuple[StochasticParticleCloud, np.ndarray]
        tuple containing updated particle cloud and per-particle incidence array.
    '''
    importation_rate = np.asarray(importation_rate, dtype=float)
    A = config['age_structure']['n_age_groups']
    P = cloud.n_particles
    t = cloud.t
    M, S, E, I, R = (cloud.counts[:, i, :] for i in range(5))

    beta_0 = np.broadcast_to(np.atleast_1d(parameters['beta_0']), (P,))
    beta_1 = np.broadcast_to(np.atleast_1d(parameters['beta_1']), (P,))
    phi = np.broadcast_to(np.atleast_1d(parameters['phi']), (P,))
    gamma = parameters['gamma']
    sigma = parameters['sigma']
    delta = parameters['maternal_waning_rate']
    birth_rate = _resolve_rate(parameters['birth_rate'], t)
    death_rate = _resolve_rate(parameters['death_rate'], t)
    v = np.asarray(_resolve_rate(parameters['vaccination_rates'], t), dtype=float)
    C = np.asarray(parameters['contact_matrix'], dtype=float)

    N = M + S + E + I + R
    with np.errstate(divide='ignore', invalid='ignore'):
        prevalence = np.where(N > 0, I / np.where(N > 0, N, 1), 0.0).astype(float)
    beta_t = beta_0 * (1.0 + beta_1 * np.cos(2 * np.pi * t + phi))
    lam = beta_t[:, None] * (prevalence @ C.T)

    g = _aging_rates()

    new_M = np.zeros((P, A), dtype=np.int64)
    new_S = np.zeros((P, A), dtype=np.int64)
    new_E = np.zeros((P, A), dtype=np.int64)
    new_I = np.zeros((P, A), dtype=np.int64)
    new_R = np.zeros((P, A), dtype=np.int64)
    incidence = np.zeros((P, A), dtype=np.int64)

    for a in range(A):
        has_next = a < A - 1
        g_a = g[a]

        chan_rates = [np.full(P, death_rate), np.full(P, delta)]
        if has_next:
            chan_rates.append(np.full(P, g_a))
        rates = np.stack(chan_rates, axis=1)
        pvals = _channel_pvals_batch(rates, dt)
        draw = _multinomial_draw_batch(M[:, a], pvals, rng)
        n_wane = draw[:, 1]
        new_M[:, a] += draw[:, -1]
        new_S[:, a] += n_wane
        if has_next:
            new_M[:, a + 1] += draw[:, 2]

        if has_next:
            rates = np.stack(
                [np.full(P, death_rate), lam[:, a], np.full(P, g_a * (1.0 - v[a])), np.full(P, g_a * v[a])],
                axis=1,
            )
        else:
            rates = np.stack([np.full(P, death_rate), lam[:, a]], axis=1)
        pvals = _channel_pvals_batch(rates, dt)
        draw = _multinomial_draw_batch(S[:, a], pvals, rng)
        n_infect = draw[:, 1]
        new_E[:, a] += n_infect
        if has_next:
            new_S[:, a + 1] += draw[:, 2]
            new_R[:, a + 1] += draw[:, 3]
        new_S[:, a] += draw[:, -1]

        chan_rates = [np.full(P, death_rate), np.full(P, sigma)]
        if has_next:
            chan_rates.append(np.full(P, g_a))
        rates = np.stack(chan_rates, axis=1)
        pvals = _channel_pvals_batch(rates, dt)
        draw = _multinomial_draw_batch(E[:, a], pvals, rng)
        n_progress = draw[:, 1]
        incidence[:, a] += n_progress
        new_I[:, a] += n_progress
        if has_next:
            new_E[:, a + 1] += draw[:, 2]
        new_E[:, a] += draw[:, -1]

        chan_rates = [np.full(P, death_rate), np.full(P, gamma)]
        if has_next:
            chan_rates.append(np.full(P, g_a))
        rates = np.stack(chan_rates, axis=1)
        pvals = _channel_pvals_batch(rates, dt)
        draw = _multinomial_draw_batch(I[:, a], pvals, rng)
        n_recover = draw[:, 1]
        new_R[:, a] += n_recover
        if has_next:
            new_I[:, a + 1] += draw[:, 2]
        new_I[:, a] += draw[:, -1]

        chan_rates = [np.full(P, death_rate)]
        if has_next:
            chan_rates.append(np.full(P, g_a))
        rates = np.stack(chan_rates, axis=1)
        pvals = _channel_pvals_batch(rates, dt)
        draw = _multinomial_draw_batch(R[:, a], pvals, rng)
        if has_next:
            new_R[:, a + 1] += draw[:, 1]
        new_R[:, a] += draw[:, -1]

    N_total = (M + S + E + I + R).sum(axis=1).astype(float)
    mean_births = birth_rate * N_total * dt
    n_births = rng.poisson(np.clip(mean_births, 0.0, None))
    new_M[:, 0] += n_births

    expected_importations = importation_rate[None, :] * dt

    n_imported = rng.poisson(
        np.broadcast_to(
            expected_importations,
            (P, A),
        )
    )

    n_imported = np.minimum(n_imported, new_S)

    new_S -= n_imported
    new_E += n_imported
    incidence += n_imported

    new_counts = np.stack([new_M, new_S, new_E, new_I, new_R], axis=1)
    return StochasticParticleCloud(t=t + dt, counts=new_counts), incidence


def simulate_interval_batch(cloud: StochasticParticleCloud, duration: float, parameters: dict[str, Any], rng: np.random.Generator, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> tuple[StochasticParticleCloud, np.ndarray]:
    '''batch interval simulation

    Parameters
    ----------
    cloud : StochasticParticleCloud
        starting particle cloud.
    duration : float
        years to advance simulation.
    parameters : dict[str, Any]
        model parameter dictionary.
    rng : np.random.Generator
        random generator.
    dt : Optional[float], default=None
        sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate vector.

    Returns
    -------
    tuple[StochasticParticleCloud, np.ndarray]
        tuple containing final particle cloud and total incidence array.
    '''
    if dt is None:
        dt = float(config['model']['stochastic_dt_years'])
    n_steps = max(1, int(np.ceil(duration / dt)))
    actual_dt = duration / n_steps

    total_incidence = np.zeros((cloud.n_particles, config['age_structure']['n_age_groups']), dtype=np.int64)
    current = cloud
    for _ in range(n_steps):
        current, inc = euler_multinomial_step_batch(
            current, actual_dt, parameters, rng, importation_rate=importation_rate
        )
        total_incidence += inc
    return current, total_incidence


def simulate_trajectory(initial_state: StochasticState, observation_times: np.ndarray, parameters: dict[str, Any], rng: np.random.Generator, dt: Optional[float] = None, importation_rate: np.ndarray = np.asarray(config['model']['default_importation_rate'], dtype=float)) -> tuple[list[StochasticState], np.ndarray]:
    '''simulates a single realization trajectory across observation times

    Parameters
    ----------
    initial_state : StochasticState
        starting state at initial observation time.
    observation_times : np.ndarray
        strictly increasing observation times in years.
    parameters : dict[str, Any]
        model parameter dictionary.
    rng : np.random.Generator
        random generator.
    dt : Optional[float], default=None
        sub-step size in years.
    importation_rate : np.ndarray, default=config['model']['default_importation_rate']
        importation rate vector.

    Returns
    -------
    tuple[list[StochasticState], np.ndarray]
        tuple containing list of states and matrix of incidence counts.
    '''
    states = [initial_state]
    incidences = []
    current = initial_state
    for t_next in observation_times[1:]:
        duration = t_next - current.t
        current, inc = simulate_interval(
            current, duration, parameters, rng, dt=dt, importation_rate=importation_rate
        )
        states.append(current)
        incidences.append(inc)
    return states, np.array(incidences)


def _aging_rates(age_group_widths: list[Optional[float]] = config['age_structure']['age_group_widths']) -> np.ndarray:
    '''calculates per group aging out rate g_a = 1 / width_a
    Parameters
    ----------
    age_group_widths : list[Optional[float]], default=config['age_structure']['age_group_widths']
        list of age group widths in years.

    Returns
    -------
    np.ndarray
        aging-out rates array per age group.
    '''
    return np.array([1.0 / w if w is not None else 0.0 for w in age_group_widths])


def force_of_infection(M: np.ndarray, S: np.ndarray, E: np.ndarray, I: np.ndarray, R: np.ndarray, t: float, beta_0: float, beta_1: float, phi: float, contact_matrix: np.ndarray) -> np.ndarray:
    '''calculates age specific force of infection lambda_a(t) = beta(t) * sum_b C[a,b] * I_b/N_b

    Parameters
    ----------
    M : np.ndarray
        maternal immune compartment counts.
    S : np.ndarray
        susceptible compartment counts.
    E : np.ndarray
        exposed compartment counts.
    I : np.ndarray
        infectious compartment counts.
    R : np.ndarray
        recovered compartment counts.
    t : float
        simulation time in years.
    beta_0 : float
        mean transmission rate.
    beta_1 : float
        seasonal forcing amplitude.
    phi : float
        phase offset in radians.
    contact_matrix : np.ndarray
        normalized contact matrix array of shape (A, A).

    Returns
    -------
    np.ndarray
        force of infection array per age group.
    '''
    N = M + S + E + I + R
    with np.errstate(divide='ignore', invalid='ignore'):
        prevalence = np.where(N > 0, I / N, 0.0)
    beta_t = seasonal_beta(t, beta_0, beta_1, phi)
    return beta_t * contact_matrix @ prevalence


def seasonal_beta(t: float, beta_0: float, beta_1: float, phi: float) -> float:
    '''computes cosine forced transmission rate beta(t) = beta_0 * (1 + beta_1 * cos(2*pi*t + phi))
    Parameters
    ----------
    t : float
        current simulation time in years.
    beta_0 : float
        mean transmission rate.
    beta_1 : float
        forcing amplitude.
    phi : float
        phase offset in radians.

    Returns
    -------
    float
        seasonally forced transmission rate beta(t).
    '''
    return beta_0 * (1.0 + beta_1 * np.cos(2.0 * np.pi * t + phi))