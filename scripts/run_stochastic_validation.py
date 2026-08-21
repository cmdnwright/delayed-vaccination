'''
orchestrates the stochastic validation process to check the calibrated MSEIR
model against the post-vaccine Texas measles series

re-runs the calibration window particle filter at theta_hat to capture a
warm start particle cloud, then forward simulates the validation window
and scores the result with calibration/proper scoring/age stratified
metrics. 

saves the validation result to data/processed/stochastic_validation_result.json 
and the final particle cloud to data/processed/stochastic_validation_final_cloud.npz

change the validation window by editing --validation-end

run using python -m scripts.run_stochastic_validation
'''

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.model import stochastic_model
from src.data import contact_matrix, data, historical_population
from src.validation import validation, validation_metrics
from src.utils import load_config

config = load_config('config.yaml')

DEFAULT_CALIBRATION_RESULT_PATH = Path(config['paths']['data_processed_dir']) / 'stochastic_calibration_result.json'
DEFAULT_VALIDATION_RESULT_PATH = Path(config['paths']['data_processed_dir']) / 'stochastic_validation_result.json'
DEFAULT_FINAL_CLOUD_PATH = Path(config['paths']['data_processed_dir']) / 'stochastic_validation_final_cloud.npz'


def restrict_date_range(series, start: str, end: str):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (series['period_start'] >= start_ts) & (series['period_start'] < end_ts)
    restricted = series.loc[mask].reset_index(drop=True)
    if restricted.empty:
        raise ValueError(
            f'No processed case data falls within [{start}, {end}). Check that '
            'the raw Tycho export actually covers this range.'
        )
    return restricted


def _shift_date_str(date_str: str, years: float) -> str:
    ts = pd.Timestamp(date_str) + pd.Timedelta(days=365.25 * years)
    return ts.strftime('%Y-%m-%d')


def pd_year(date_str: str) -> int:
    return int(date_str.split('-')[0])


def build_fixed_parameters(fit_window_start_year: float, cfg=config) -> dict:
    birth_rate_fn, death_rate_fn = historical_population.build_vital_rate_functions()
    contact_df = contact_matrix.load_processed_contact_matrix(cfg)
    return {
        'gamma': cfg['calibration']['calibration_fixed_parameters']['gamma'],
        'sigma': cfg['calibration']['calibration_fixed_parameters']['sigma'],
        'maternal_waning_rate': cfg['calibration']['calibration_fixed_parameters']['maternal_waning_rate'],
        'birth_rate': birth_rate_fn,
        'death_rate': death_rate_fn,
        'vaccination_rates': list(cfg['calibration']['calibration_fixed_vaccination_rates']),
        'contact_matrix': contact_df.to_numpy(dtype=float),
    }


def main(
    calibration_result_path: Path = DEFAULT_CALIBRATION_RESULT_PATH,
    validation_end: str = '2000-01-01',
    result_path: Path = DEFAULT_VALIDATION_RESULT_PATH,
    final_cloud_path: Path = DEFAULT_FINAL_CLOUD_PATH,
    warm_start_n_particles: int = config['calibration']['particle_filter_n_particles_final'],
    warm_start_seed: int = 1,
    validation_seed: int = 2,
) -> dict:
    with open(calibration_result_path) as f:
        calib_result = json.load(f)

    theta_hat = calib_result['theta_hat']
    calib_start = calib_result['fit_window']['start']
    calib_end = calib_result['fit_window']['end']
    spin_up_years = calib_result.get('spin_up_years', 0.0)

    series = data.load_processed_series()

    calib_sim_start = _shift_date_str(calib_start, -spin_up_years) if spin_up_years > 0 else calib_start
    calib_fit_series = restrict_date_range(series, calib_sim_start, calib_end)
    calib_observed_times = data.series_to_elapsed_years(calib_fit_series)
    calib_observed_case_counts = calib_fit_series['case_count'].to_numpy(dtype=float)
    fit_window_start_year = float(pd_year(calib_sim_start))
    score_start = spin_up_years if spin_up_years > 0 else None

    calibration_fixed_parameters = build_fixed_parameters(fit_window_start_year)

    print(f're-running calibration window particle filter at theta_hat with {warm_start_n_particles} particles')
    warm_start_result = validation.capture_calibration_final_cloud(
        theta_hat=theta_hat,
        observed_times=calib_observed_times,
        observed_case_counts=calib_observed_case_counts,
        fixed_parameters=calibration_fixed_parameters,
        fit_window_start_year=fit_window_start_year,
        n_particles=warm_start_n_particles,
        seed=warm_start_seed,
        score_start=score_start,
    )
    calibration_final_cloud = stochastic_model.StochasticParticleCloud.from_states(
        warm_start_result.final_states
    )

    validation_start_year = float(pd_year(calib_end))
    validation_series = restrict_date_range(series, calib_end, validation_end)
    validation_observed_times = data.series_to_elapsed_years(validation_series)
    validation_observed_case_counts = validation_series['case_count'].to_numpy(dtype=float)

    print(f'forward-simulating validation window {calib_end} to {validation_end}')
    validation_fixed_parameters = validation.build_validation_fixed_parameters(
        calibration_fixed_parameters, validation_start_year=validation_start_year,
    )
    rng = np.random.default_rng(validation_seed)
    result = validation.run_validation(
        theta_hat=theta_hat,
        calibration_final_cloud=calibration_final_cloud,
        validation_observed_times=validation_observed_times,
        validation_observed_case_counts=validation_observed_case_counts,
        validation_fixed_parameters=validation_fixed_parameters,
        validation_start_year=validation_start_year,
        rng=rng,
        return_final_cloud=True
    )

    if result.final_particle_cloud is None:
        raise RuntimeError('validation did not return its final particle cloud')

    print('computing calibration, proper-scoring, and age-stratified metrics')
    metrics_rng = np.random.default_rng(validation_seed + 1000)
    metrics = validation_metrics.compute_all_metrics(
        case_draws=result.simulated_case_draws,
        validation_times=result.validation_times,
        observed_case_counts=result.observed_case_counts,
        simulated_incidence=result.simulated_incidence,
        pit_rng=metrics_rng,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    final_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        final_cloud_path,
        t=np.array(result.final_particle_cloud.t),
        counts=result.final_particle_cloud.counts,
    )
    payload = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'calibration_result_path': str(calibration_result_path),
        'calibration_fit_window': {'start': calib_start, 'end': calib_end},
        'validation_window': {'start': calib_end, 'end': validation_end},
        'vaccination_ramp_start': calib_end,
        'theta_hat': theta_hat,
        'warm_start_n_particles': warm_start_n_particles,
        'validation_times': result.validation_times.tolist(),
        'observed_case_counts': result.observed_case_counts.tolist(),
        'simulated_case_counts_mean': result.simulated_case_counts_mean.tolist(),
        'simulated_case_counts_ci_5_95': result.simulated_case_counts_ci.tolist(),
        'log_predictive_density': result.log_predictive_density,
        'n_particles': result.n_particles,
        'final_particle_cloud_path': str(final_cloud_path),
        'final_particle_cloud_shape': list(result.final_particle_cloud.counts.shape),
        'metrics': metrics,
    }
    with open(result_path, 'w') as f:
        json.dump(payload, f, indent=2)

    return {'result': result, 'payload': payload}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-result', default=str(DEFAULT_CALIBRATION_RESULT_PATH))
    parser.add_argument('--validation-end', default='2000-01-01')
    parser.add_argument('--result-path', default=str(DEFAULT_VALIDATION_RESULT_PATH))
    parser.add_argument('--final-cloud-path', default=str(DEFAULT_FINAL_CLOUD_PATH))
    parser.add_argument('--warm-start-n-particles', type=int, default=config['calibration']['particle_filter_n_particles_final'])
    parser.add_argument('--warm-start-seed', type=int, default=1)
    parser.add_argument('--validation-seed', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(
        calibration_result_path=Path(args.calibration_result),
        validation_end=args.validation_end,
        result_path=Path(args.result_path),
        final_cloud_path=Path(args.final_cloud_path),
        warm_start_n_particles=args.warm_start_n_particles,
        warm_start_seed=args.warm_start_seed,
        validation_seed=args.validation_seed,
    )