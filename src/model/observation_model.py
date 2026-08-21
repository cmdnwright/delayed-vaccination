import numpy as np
from scipy import stats


def _nb_n_p(mu: np.ndarray, phi_obs: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''converts mu and phi obs to n p parameterization as expected by negative binomial

    Parameters
    ----------
    mu : np.ndarray
        mean expected reported case count array.
    phi_obs : float
        observation dispersion parameter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        converted (n, p) parameterization tuple for scipy.
    '''
    mu = np.asarray(mu, dtype=float)
    n = np.full_like(mu, phi_obs, dtype=float)
    p = phi_obs / (phi_obs + np.clip(mu, 1e-12, None))
    return n, p


def negbin_logpmf(y: np.ndarray, mu: np.ndarray, phi_obs: float | np.ndarray) -> np.ndarray:
    ''' computes log(P(Y_t = y | mu_t, phi_obs)) elementwise
    Parameters
    ----------
    y : np.ndarray
        observed case count array.
    mu : np.ndarray
        simulated incidence mean array.
    phi_obs : float
        observation dispersion parameter.

    Returns
    -------
    np.ndarray
        log probability mass function values.
    '''
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    out = np.full(np.broadcast_shapes(y.shape, mu.shape), -np.inf, dtype=float)
    zero_mu = mu <= 0
    out = np.where(zero_mu, np.where(y == 0, 0.0, -np.inf), out)
    if np.any(~zero_mu):
        n, p = _nb_n_p(np.where(zero_mu, 1.0, mu), phi_obs)
        logpmf = stats.nbinom.logpmf(y, n, p)
        out = np.where(zero_mu, out, logpmf)
    return out


def negbin_rvs(mu: np.ndarray, phi_obs: float | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''samples negative binomial model for posterior predictive checks

    Parameters
    ----------
    mu : np.ndarray
        simulated incidence mean array.
    phi_obs : float
        observation dispersion parameter.
    rng : np.random.Generator
        random number generator.

    Returns
    -------
    np.ndarray
        sampled observation counts array.
    '''
    mu = np.asarray(mu, dtype=float)
    out = np.zeros_like(mu)
    nonzero = mu > 0
    if np.any(nonzero):
        n, p = _nb_n_p(mu[nonzero], phi_obs)
        out[nonzero] = rng.negative_binomial(n, p)
    return out