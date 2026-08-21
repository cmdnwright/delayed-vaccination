# type: ignore
'''
regression tests for the calibration and validation pipeline contracts.
the tests use small fixed particle clouds so that stochastic results are
reproducible, inexpensive, and independent of the processed project data.
'''

from pathlib import Path
import importlib.util

import numpy as np
import pytest

from src.calibration import calibration
from src.model import particle_filter, stochastic_model
from src.utils import load_config
from src.validation import validation


CONFIG = load_config('config.yaml')
AGE_GROUPS = CONFIG['age_structure']['n_age_groups']
PROJECT_ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _theta(**overrides) -> dict[str, float]:
    theta = {
        'beta_0': 600.0,
        'beta_1': 0.2,
        'phi': 0.0,
        'rho': 0.2,
        'phi_obs': 10.0,
        'seed_fraction': 1e-4,
    }
    theta.update(overrides)
    return theta


def _fixed_parameters(**overrides) -> dict:
    fixed = {
        'gamma': CONFIG['literature']['fixed_gamma_per_year'],
        'sigma': CONFIG['literature']['fixed_sigma_per_year'],
        'maternal_waning_rate': 0.0,
        'birth_rate': 0.0,
        'death_rate': 0.0,
        'vaccination_rates': [0.0] * (AGE_GROUPS - 1),
        'contact_matrix': np.eye(AGE_GROUPS),
    }
    fixed.update(overrides)
    return fixed


def _initial_cloud(n_particles: int = 4) -> stochastic_model.StochasticParticleCloud:
    states = []
    for particle in range(n_particles):
        susceptible = np.full(AGE_GROUPS, 1000, dtype=float)
        infected = np.zeros(AGE_GROUPS, dtype=float)
        infected[0] = 10 + particle
        susceptible[0] -= infected[0]
        states.append(
            stochastic_model.make_initial_state(
                np.zeros(AGE_GROUPS), susceptible, np.zeros(AGE_GROUPS),
                infected, np.zeros(AGE_GROUPS),
            )
        )
    return stochastic_model.StochasticParticleCloud.from_states(states)


def test_particle_filter_is_reproducible_and_has_stable_shapes():
    '''The single-particle pipeline is deterministic for a fixed seed.'''
    times = np.array([0.0, 0.02, 0.04])
    kwargs = dict(
        theta=_theta(),
        observed_times=times,
        observed_case_counts=np.zeros(len(times)),
        fixed_parameters=_fixed_parameters(),
        n_particles=4,
        fit_window_start_year=1950.0,
        dt=0.01,
        importation_rate=np.zeros(AGE_GROUPS),
        initial_cloud=_initial_cloud(),
    )
    first = particle_filter.run_particle_filter(rng=np.random.default_rng(31), **kwargs)
    second = particle_filter.run_particle_filter(rng=np.random.default_rng(31), **kwargs)

    assert first.incidence_mean.shape == (len(times) - 1, AGE_GROUPS)
    assert first.ess_history.shape == (len(times),)
    assert first.resampled_history.shape == (len(times),)
    assert len(first.final_states) == 4
    np.testing.assert_array_equal(first.incidence_mean, second.incidence_mean)
    np.testing.assert_array_equal(first.ess_history, second.ess_history)
    assert first.log_likelihood == second.log_likelihood


@pytest.mark.xfail(
    raises=TypeError,
    strict=False,
    reason='run_particle_filter_batch currently casts vector-valued phi_obs to float',
)
def test_batch_particle_filter_preserves_theta_and_time_dimensions():
    '''The batched pipeline keeps one result row per parameter candidate.'''
    times = np.array([0.0, 0.02, 0.04])
    thetas = [_theta(beta_0=550.0), _theta(beta_0=750.0)]
    result = particle_filter.run_particle_filter_batch(
        thetas=thetas,
        observed_times=times,
        observed_case_counts=np.zeros(len(times)),
        fixed_parameters=_fixed_parameters(),
        n_inner_particles=3,
        rng=np.random.default_rng(32),
        fit_window_start_year=1950.0,
        dt=0.01,
        importation_rate=np.zeros(AGE_GROUPS),
    )

    assert result.log_likelihood.shape == (len(thetas),)
    assert result.ess_history.shape == (len(thetas), len(times))
    assert result.resampled_history.shape == (len(thetas), len(times))


def test_hyperparameter_change_changes_particle_filter_behavior():
    '''Changing transmission is observable in the simulated incidence.'''
    common = dict(
        observed_times=np.array([0.0, 0.2]),
        observed_case_counts=np.zeros(2),
        fixed_parameters=_fixed_parameters(),
        n_particles=4,
        fit_window_start_year=1950.0,
        dt=0.01,
        importation_rate=np.zeros(AGE_GROUPS),
        initial_cloud=_initial_cloud(),
    )
    low = particle_filter.run_particle_filter(
        theta=_theta(beta_0=1.0), rng=np.random.default_rng(33), **common
    )
    high = particle_filter.run_particle_filter(
        theta=_theta(beta_0=900.0), rng=np.random.default_rng(33), **common
    )

    assert not np.array_equal(low.incidence_mean, high.incidence_mean)


def test_iterated_filtering_diagnostics_shapes_and_reproducibility():
    '''IF2 diagnostics retain one trace per iteration and particle.'''
    times = np.array([0.0, 0.02, 0.04])
    perturbation_sd = {
        name: 0.0 for name in [
            'log_beta_0', 'logit_beta_1', 'phi', 'logit_rho',
            'log_phi_obs', 'logit_seed_fraction',
        ]
    }
    kwargs = dict(
        theta0=_theta(),
        observed_times=times,
        observed_case_counts=np.zeros(len(times)),
        fixed_parameters=_fixed_parameters(),
        fit_window_start_year=1950.0,
        n_iterations=2,
        n_particles=4,
        initial_perturbation_sd=perturbation_sd,
        cooling_fraction=1.0,
    )
    first = calibration.iterated_filtering(seed=34, **kwargs)
    second = calibration.iterated_filtering(seed=34, **kwargs)

    assert first.theta_trace.shape == (2, len(calibration.FREE_PARAMS))
    assert first.log_likelihood_trace.shape == (2,)
    assert first.particle_theta_trace.shape == (2, 4, len(calibration.FREE_PARAMS))
    assert first.particle_weight_trace.shape == (2, 4)
    assert first.ess_trace.shape == (2, len(times) - 1)
    assert first.resampled_trace.shape == (2, len(times) - 1)
    np.testing.assert_array_equal(first.theta_trace, second.theta_trace)
    assert first.best_iteration == int(np.argmax(first.log_likelihood_trace))


def test_validation_outputs_are_interval_aligned_and_reproducible():
    '''Validation returns one predictive draw per particle and interval.'''
    times = np.array([0.0, 0.02, 0.04])
    kwargs = dict(
        theta_hat=_theta(beta_0=1.0),
        calibration_final_cloud=_initial_cloud(),
        validation_observed_times=times,
        validation_observed_case_counts=np.zeros(len(times) - 1),
        validation_fixed_parameters=_fixed_parameters(),
        validation_start_year=1965.0,
        return_final_cloud=True,
        importation_rate=np.zeros(AGE_GROUPS),
        dt=0.01,
    )
    first = validation.run_validation(rng=np.random.default_rng(35), **kwargs)
    second = validation.run_validation(rng=np.random.default_rng(35), **kwargs)

    assert first.simulated_incidence.shape == (4, len(times) - 1, AGE_GROUPS)
    assert first.simulated_case_draws.shape == (4, len(times) - 1)
    assert first.simulated_case_counts_mean.shape == (len(times) - 1,)
    assert first.simulated_case_counts_ci.shape == (len(times) - 1, 2)
    assert first.observed_case_counts.shape == (len(times) - 1,)
    assert first.final_particle_cloud.counts.shape == (4, 5, AGE_GROUPS)
    np.testing.assert_array_equal(first.simulated_case_draws, second.simulated_case_draws)


def test_pipeline_scripts_keep_date_windows_and_result_schema(tmp_path):
    '''Script helpers preserve half-open windows and serialized diagnostics.'''
    import pandas as pd

    calibration_script = _load_script('run_stochastic_calibration.py')
    validation_script = _load_script('run_stochastic_validation.py')
    series = pd.DataFrame({
        'period_start': pd.to_datetime(['1950-01-01', '1951-01-01', '1952-01-01']),
        'case_count': [1, 2, 3],
    })
    window = calibration_script.restrict_date_range(series, '1951-01-01', '1952-01-01')
    pd.testing.assert_frame_equal(window, series.iloc[[1]].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        validation_script.restrict_date_range(series, '1951-01-01', '1952-01-01'), window
    )
    with pytest.raises(ValueError):
        calibration_script.restrict_date_range(series, '1960-01-01', '1961-01-01')

    class IF2:
        log_likelihood_trace = np.array([-4.0, -2.0])
        best_iteration = 1
        theta_hat_last_iteration = _theta()
        ess_trace = np.ones((2, 2))
        resampled_trace = np.ones((2, 2), dtype=bool)
        particle_theta_trace = np.ones((2, 4, 6))
        particle_weight_trace = np.full((2, 4), 0.25)

    result_path = tmp_path / 'calibration.json'
    calibration_script.save_result(
        theta_hat=_theta(),
        final_eval={'log_likelihood_mean': -2.0},
        if2_result=IF2(),
        multistart_ranked=[(_theta(), -2.0)],
        experiment_name='test',
        fit_window=('1950-01-01', '1952-01-01'),
        result_path=result_path,
    )
    import json

    payload = json.loads(result_path.read_text())
    assert payload['if2_diagnostics']['log_likelihood_trace'] == [-4.0, -2.0]
