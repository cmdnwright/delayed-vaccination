import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.model import particle_filter 

FREE_PARAM_LABELS = {
    'beta_0': r'$\beta_0$ (mean transmission)',
    'beta_1': r'$\beta_1$ (seasonal amplitude)',
    'phi': r'$\phi$ (seasonal phase)',
    'rho': r'$\rho$ (reporting probability)',
    'phi_obs': r'$\phi_{obs}$ (neg-binom overdispersion)',
    'seed_fraction': 'seed_fraction',
}


def load_result(result_path: Path) -> dict:
    '''loads a saved calibration result JSON

    Parameters
    ----------
    result_path : Path
        path to the JSON file written by run_stochastic_calibration.save_result.

    Returns
    -------
    dict
        the parsed result, including the if2_diagnostics block if present.
    '''
    with open(result_path) as f:
        return json.load(f)


def _require_diagnostics(result: dict) -> dict:
    '''pulls the if2_diagnostics block out of a result, failing clearly on old result files

    Parameters
    ----------
    result : dict
        a result loaded via load_result.

    Returns
    -------
    dict
        the if2_diagnostics block.
    '''
    diag = result.get('if2_diagnostics')
    if diag is None:
        raise ValueError('result file has no if2_diagnostics block')
    return diag


def parameter_path_frame(result: dict) -> pd.DataFrame:
    '''one row per (iteration, free parameter): weighted mean, weighted SD (i.e. the spread of
    that iteration's perturbed-particle cloud around the mean), and the iteration's
    log-likelihood, for line+band plots

    Parameters
    ----------
    result : dict
        a result loaded via load_result, containing an if2_diagnostics block.

    Returns
    -------
    pd.DataFrame
        columns: iteration, parameter, weighted_mean, weighted_sd,
        log_likelihood, is_best_iteration.
    '''
    diag = _require_diagnostics(result)
    free_params = diag['free_param_order']
    particle_theta = np.array(diag['particle_theta_trace']) # (n_iter, n_particles, 6)
    weights = np.array(diag['particle_weight_trace']) # (n_iter, n_particles)
    loglik = np.array(diag['log_likelihood_trace'])

    n_iter = particle_theta.shape[0]
    rows = []
    for it in range(n_iter):
        w = weights[it]
        for j, name in enumerate(free_params):
            vals = particle_theta[it, :, j]
            mean = float(np.sum(w * vals))
            var = float(np.sum(w * (vals - mean) ** 2))
            rows.append({
                'iteration': it,
                'parameter': name,
                'weighted_mean': mean,
                'weighted_sd': np.sqrt(max(var, 0.0)),
                'log_likelihood': float(loglik[it]),
                'is_best_iteration': it == diag['best_iteration'],
            })
    return pd.DataFrame(rows)


def plot_parameter_paths(result: dict, log_scale_params=('beta_0', 'phi_obs', 'seed_fraction')):
    '''grid of small multiples: one subplot per free parameter, weighted mean +/- 1 weighted SD
    per iteration, best iteration marked

    Parameters
    ----------
    result : dict
        a result loaded via load_result, containing an if2_diagnostics block.
    log_scale_params : tuple, optional
        free parameter names to plot on a log y-axis, by default
        ('beta_0', 'phi_obs', 'seed_fraction').

    Returns
    -------
    matplotlib.figure.Figure
        the parameter-path grid.
    '''
    import matplotlib.pyplot as plt

    diag = _require_diagnostics(result)
    free_params = diag['free_param_order']
    df = parameter_path_frame(result)
    best_it = diag['best_iteration']

    n = len(free_params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

    for i, name in enumerate(free_params):
        ax = axes[i // ncols][i % ncols]
        sub = df[df['parameter'] == name].sort_values('iteration')
        mean = sub['weighted_mean'].to_numpy()
        sd = sub['weighted_sd'].to_numpy()
        it = sub['iteration'].to_numpy()

        ax.plot(it, mean, color='C0', marker='o', ms=3)
        ax.fill_between(it, mean - sd, mean + sd, color='C0', alpha=0.2)
        ax.axvline(best_it, color='red', ls='--', lw=1, label='best iteration')
        ax.set_title(FREE_PARAM_LABELS.get(name, name))
        ax.set_xlabel('IF2 iteration')
        if name in log_scale_params:
            ax.set_yscale('log')

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    axes[0][0].legend(loc='best', fontsize=8)
    fig.suptitle('IF2 parameter paths (weighted mean \u00b1 1 SD of perturbed-particle cloud)')
    fig.tight_layout()
    return fig


def plot_loglik_and_ess_trace(result: dict):
    '''plots the IF2 log-likelihood trace and the sequential ESS trace. the saved IF2
    diagnostics contain log_likelihood_trace with shape (n_iterations,) and ess_trace with
    shape (n_iterations, n_observation_intervals), so the ESS panel shows particle-cloud
    degeneracy across both IF2 iteration and observation interval

    Parameters
    ----------
    result : dict
        a result loaded via load_result, containing an if2_diagnostics block.

    Returns
    -------
    matplotlib.figure.Figure
        two-panel figure: log-likelihood trace on top, ESS/N heatmap below.
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    diag = _require_diagnostics(result)

    loglik = np.asarray(diag['log_likelihood_trace'], dtype=float)
    ess = np.asarray(diag['ess_trace'], dtype=float)
    best_it = int(diag['best_iteration'])

    particle_theta_trace = np.asarray(diag['particle_theta_trace'])
    n_particles = particle_theta_trace.shape[1]

    # validate the diagnostic shapes explicitly
    if ess.ndim != 2:
        raise ValueError(f'ess shape {ess.shape}. failed')

    n_iterations, n_intervals = ess.shape

    if len(loglik) != n_iterations:
        raise ValueError('log_likelihood_trace and ess_trace incompatible')

    it = np.arange(n_iterations)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        gridspec_kw={'height_ratios': [1, 2]},
    )

    # panel 1: IF2 log-likelihood trace
    finite = np.isfinite(loglik)

    if np.any(finite):
        ax1.plot(
            it[finite],
            loglik[finite],
            marker='o',
            ms=3,
            label='log-likelihood + log-prior',
        )

    if np.any(~finite):
        if np.any(finite):
            y_bad = np.nanmin(loglik[finite])
            y_bad = y_bad - 0.05 * max(abs(y_bad), 1.0)
        else:
            y_bad = 0.0

        ax1.scatter(
            it[~finite],
            np.full(np.sum(~finite), y_bad),
            marker='x',
            label='non-finite',
        )

    if 0 <= best_it < n_iterations:
        ax1.axvline(
            best_it,
            linestyle='--',
            linewidth=1,
            label=f'chosen iteration = {best_it}',
        )

    ax1.set_ylabel('log-likelihood + log-prior')
    ax1.set_title('IF2 log-likelihood trace')
    ax1.legend(fontsize=8)

    ess_fraction = ess / float(n_particles)

    image = ax2.imshow(
        ess_fraction,
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        extent=[
            0,
            n_intervals - 1,
            0,
            n_iterations - 1,
        ],
        vmin=0.0,
        vmax=1.0,
    )

    ax2.axhline(
        best_it,
        linestyle='--',
        linewidth=1,
        label=f'chosen iteration = {best_it}',
    )

    ax2.set_ylabel('IF2 iteration')
    ax2.set_xlabel('observation interval')
    ax2.set_title(
        f'sequential particle ESS / N '
        f'(N = {n_particles})'
    )

    cbar = fig.colorbar(image, ax=ax2)
    cbar.set_label('ESS / N')

    ax2.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    return fig


def weighted_correlation_matrix(result: dict, iteration: Optional[int] = None) -> pd.DataFrame:
    '''weighted pearson correlation between free parameters, using one iteration's
    perturbed-particle cloud and its resampling weights. defaults to the best-seen iteration
    (the one theta_hat came from) -- that's the cloud whose correlations are most relevant to
    understanding theta_hat's uncertainty/identifiability, since it's the iteration closest
    to convergence IF2 actually reached

    Parameters
    ----------
    result : dict
        a result loaded via load_result, containing an if2_diagnostics block.
    iteration : int, optional
        which IF2 iteration's particle cloud to use, by default None
        (the best-seen iteration).

    Returns
    -------
    pd.DataFrame
        6x6 weighted correlation matrix, indexed and columned by free
        parameter name.
    '''
    diag = _require_diagnostics(result)
    if iteration is None:
        iteration = diag['best_iteration']
    free_params = diag['free_param_order']
    vals = np.array(diag['particle_theta_trace'])[iteration]  # (n_particles, 6)
    w = np.array(diag['particle_weight_trace'])[iteration]
    w = w / w.sum()

    mean = (w[:, None] * vals).sum(axis=0)
    centered = vals - mean
    cov = (w[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(sd, sd)
    return pd.DataFrame(corr, index=free_params, columns=free_params)


def plot_parameter_correlations(result: dict, iteration: Optional[int] = None):
    '''heatmap of weighted_correlation_matrix

    Parameters
    ----------
    result : dict
        a result loaded via load_result, containing an if2_diagnostics block.
    iteration : int, optional
        which IF2 iteration's particle cloud to use, by default None
        (the best-seen iteration).

    Returns
    -------
    matplotlib.figure.Figure
        the correlation heatmap.
    '''
    import matplotlib.pyplot as plt

    corr = weighted_correlation_matrix(result, iteration)
    diag = _require_diagnostics(result)
    it = diag['best_iteration'] if iteration is None else iteration

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(f'weighted parameter correlations, iteration {it}\n(perturbed-particle cloud IF2 explored around theta_hat)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='weighted correlation')
    fig.tight_layout()
    return fig


def repeated_high_particle_runs(
    theta_hat: dict,
    observed_times: np.ndarray,
    observed_case_counts: np.ndarray,
    fixed_parameters: dict,
    fit_window_start_year: float,
    n_particles: int,
    n_replicates: int,
    seed: int = 12345,
):
    '''thin wrapper around particle_filter.run_particle_filter_batch_parallel that runs many
    independent replicates at a fixed theta (typically theta_hat) and returns a DataFrame of
    per-replicate log-likelihoods plus a summary (mean, SE, extinction frequency). this is
    deliberately the same call final_likelihood_evaluation makes with n_replicates=5 -- this
    version is meant to be run with more replicates (e.g. 30-50) purely for diagnostics, to
    get a less noisy read on the extinction frequency than 5 replicates give

    Parameters
    ----------
    theta_hat : dict
        parameter point to evaluate, natural scale.
    observed_times : np.ndarray
        observation times.
    observed_case_counts : np.ndarray
        observed case counts at each observed time.
    fixed_parameters : dict
        model parameters held fixed for this calibration run.
    fit_window_start_year : float
        calendar year corresponding to observed_times[0], used to wrap
        calendar-indexed vital rate functions.
    n_particles : int
        particle count per replicate.
    n_replicates : int
        number of independent particle-filter replicates.
    seed : int, optional
        seed for the RNG controlling the per-replicate run RNG, by default 12345.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        per-replicate log-likelihoods, and a summary dict with keys
        n_particles, n_replicates, n_extinct, extinction_frequency,
        log_likelihood_mean, log_likelihood_se.
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
        verbose=5,
    )
    logliks = batch_result.log_likelihood
    finite = logliks[np.isfinite(logliks)]
    summary = {
        'n_particles': n_particles,
        'n_replicates': n_replicates,
        'n_extinct': int(n_replicates - len(finite)),
        'extinction_frequency': float((n_replicates - len(finite)) / n_replicates),
        'log_likelihood_mean': float(finite.mean()) if len(finite) else float('-inf'),
        'log_likelihood_se': float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else None,
    }
    df = pd.DataFrame({'replicate': np.arange(n_replicates), 'log_likelihood': logliks})
    return df, summary


def plot_replicate_loglik_distribution(df: pd.DataFrame, summary: dict):
    '''histogram of per-replicate log-likelihoods from repeated_high_particle_runs, with the
    finite-replicate mean and the extinction frequency annotated

    Parameters
    ----------
    df : pd.DataFrame
        per-replicate log-likelihoods, as returned by repeated_high_particle_runs.
    summary : dict
        summary dict, as returned by repeated_high_particle_runs.

    Returns
    -------
    matplotlib.figure.Figure
        the log-likelihood histogram.
    '''
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    finite = df['log_likelihood'].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(finite, bins=min(15, max(5, len(finite) // 2)), color='C0', alpha=0.8)
    if summary['log_likelihood_mean'] != float('-inf'):
        ax.axvline(summary['log_likelihood_mean'], color='red', ls='--',
                    label=f"mean = {summary['log_likelihood_mean']:.1f}")
    ax.set_xlabel('log-likelihood')
    ax.set_ylabel('replicate count')
    ax.set_title(
        f'repeated high-particle log-likelihood at theta_hat\n'
        f"extinction frequency: {summary['extinction_frequency']:.1%} "
        f"({summary['n_extinct']}/{summary['n_replicates']})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig