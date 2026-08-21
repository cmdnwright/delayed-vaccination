'''
orchestrates the stochastic calibration process to fit the MSEIR model to the
pre-vaccine Texas measles series

runs a multistart screen to find a good initial theta, refines it with
iterated filtering (IF2), and finishes with a high particle likelihood
evaluation at the resulting theta_hat. 

saves theta_hat, diagnostics, and multistart candidates to 
data/processed/stochastic_calibration_result.json

change the fit window and spin-up by editing --start / --end / --spin-up-years

run using python -m scripts.run_stochastic_calibration
'''

import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.calibration import calibration
from src.data import contact_matrix, data, historical_population
from src.utils import load_config

config = load_config('config.yaml')


DEFAULT_RESULT_PATH = Path(config['paths']['data_processed_dir']) / 'stochastic_calibration_result.json'


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


def build_fixed_parameters(fit_window_start_year: float, cfg=config) -> dict:
    '''fixed (non-estimated) parameters for the pre-vaccine calibration window
    '''
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


def save_result(
    theta_hat: dict,
    final_eval: dict,
    if2_result,
    multistart_ranked: list,
    fit_window: tuple[str, str],
    result_path: Path = DEFAULT_RESULT_PATH,
    spin_up_years: float = 0.0,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'fit_window': {'start': fit_window[0], 'end': fit_window[1]},
        'spin_up_years': spin_up_years,
        'theta_hat': theta_hat,
        'final_likelihood_evaluation': final_eval,
        'if2_diagnostics': {
            'free_param_order': list(calibration.FREE_PARAMS),
            'log_likelihood_trace': if2_result.log_likelihood_trace.tolist(),
            'best_iteration': if2_result.best_iteration,
            'n_iterations': len(if2_result.log_likelihood_trace),
            'theta_hat_last_iteration': if2_result.theta_hat_last_iteration,
            'ess_trace': if2_result.ess_trace.tolist(),
            'resampled_trace': if2_result.resampled_trace.tolist(),
            'particle_theta_trace': if2_result.particle_theta_trace.tolist(),
            'particle_weight_trace': if2_result.particle_weight_trace.tolist(),
        },
        'multistart_top_candidates': [
            {'theta': theta, 'log_posterior': log_post}
            for theta, log_post in multistart_ranked[:5]
        ],
        'fixed_parameters_summary': {
            'gamma': config['calibration']['calibration_fixed_parameters']['gamma'],
            'sigma': config['calibration']['calibration_fixed_parameters']['sigma'],
            'maternal_waning_rate': config['calibration']['calibration_fixed_parameters']['maternal_waning_rate'],
            'vaccination_rates': list(config['calibration']['calibration_fixed_vaccination_rates']),
        },
        'architecture': 'stochastic MSEIR + sequential joint state/parameter IF2 + final particle-filter evaluation',
    }
    with open(result_path, 'w') as f:
        json.dump(payload, f, indent=2)


def _shift_date_str(date_str: str, years: float) -> str:
    '''shift an ISO date string back/forward by `years` (365.25-day years,
    consistent with every other annualized quantity'''
    import pandas as pd

    ts = pd.Timestamp(date_str) + pd.Timedelta(days=365.25 * years)
    return ts.strftime('%Y-%m-%d')


def main(
    start: str = '1920-01-01',
    end: str = '1965-01-01',
    result_path: Path = DEFAULT_RESULT_PATH,
    rebuild_data: bool = True,
    spin_up_years: float = 0.0,
) -> dict:
    if rebuild_data:
        data.build_texas_measles_series()
        contact_matrix.build_processed_contact_matrix()

    series = data.load_processed_series()
    if spin_up_years > 0:
        sim_start = _shift_date_str(start, -spin_up_years)
    else:
        sim_start = start
    fit_series = restrict_date_range(series, sim_start, end)
    observed_times = data.series_to_elapsed_years(fit_series)
    observed_case_counts = fit_series['case_count'].to_numpy(dtype=float)

    fit_window_start_year = float(pd_year(sim_start))
    score_start = spin_up_years if spin_up_years > 0 else None
    fixed_parameters = build_fixed_parameters(fit_window_start_year)

    print(f"multistart screening with {config['calibration']['multistart_n_starts']} starts, {config['calibration']['multistart_n_particles']} particles each" + (f', {spin_up_years:.1f}y spin-up before {start}' if spin_up_years > 0 else ''))
    ranked = calibration.multistart_screen(
        observed_times, observed_case_counts, fixed_parameters, fit_window_start_year,
        score_start=score_start,
    )
    best_theta0, best_log_post = ranked[0]
    print(f'best multistart log-posterior: {best_log_post:.2f}')

    print(f"iterated filtering with {config['calibration']['if2_n_iterations']} iterations, {config['calibration']['if2_n_particles']} particles")
    if2_result = calibration.iterated_filtering(
        theta0=best_theta0,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        fit_window_start_year=fit_window_start_year,
        score_start=score_start,
    )

    print(f"high-particle likelihood evaluation with {config['calibration']['particle_filter_n_particles_final']} particles")
    final_eval = calibration.final_likelihood_evaluation(
        theta_hat=if2_result.theta_hat,
        observed_times=observed_times,
        observed_case_counts=observed_case_counts,
        fixed_parameters=fixed_parameters,
        fit_window_start_year=fit_window_start_year,
        score_start=score_start,
    )

    save_result(
        theta_hat=if2_result.theta_hat,
        final_eval=final_eval,
        if2_result=if2_result,
        multistart_ranked=ranked,
        fit_window=(start, end),
        result_path=result_path,
        spin_up_years=spin_up_years,
    )

    return {
        'theta_hat': if2_result.theta_hat,
        'final_likelihood_evaluation': final_eval,
        'if2_result': if2_result,
    }


def pd_year(date_str: str) -> int:
    return int(date_str.split('-')[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='1950-01-01')
    parser.add_argument('--end', default='1965-01-01')
    parser.add_argument('--result-path', default=str(DEFAULT_RESULT_PATH))
    parser.add_argument('--no-rebuild-data', action='store_true')
    parser.add_argument('--spin-up-years', type=float, default=config['calibration']['spin_up_years'],)
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(
        start=args.start,
        end=args.end,
        result_path=Path(args.result_path),
        rebuild_data=not args.no_rebuild_data,
        spin_up_years=args.spin_up_years,
    )