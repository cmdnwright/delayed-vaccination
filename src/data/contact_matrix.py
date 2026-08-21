from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.utils import load_config

config = load_config('config.yaml')


def load_raw_contact_matrix(path: Path = config['contact_matrix']['raw_contact_matrix_path'], band_labels: Optional[list[str]] = None) -> pd.DataFrame:
    '''loads original contact matrix

    Parameters
    ----------
    path : Path
        CSV with a square matrix and a header row of band labels, but no
        row-label column -- the matrix is symmetric, so the column
        headers double as the row labels.
    band_labels : list of str, optional
        expected band labels, for a sanity check against the file's
        header, and used as the row labels (the file itself has none).
        defaults to `config['contact_matrix']['original_contact_band_labels'].

    Returns
    -------
    pd.DataFrame
        square DataFrame indexed and columned by original band label.
    '''
    band_labels = band_labels or config['contact_matrix']['original_contact_band_labels']
    matrix = pd.read_csv(path)
    if band_labels is not None:
        assert list(matrix.columns) == list(band_labels), ('raw contact matrix column headers do not match')
    matrix.index = matrix.columns
    return matrix


def parse_nhgis_codebook(codebook_path: Path) -> dict[str, str]:
    '''parses the nhgis population breakdown code book of format A38AA2020: 2020: Persons: Under 1 year
    into ductionary from code to age label

    Parameters
    ----------
    codebook_path : Path
        path to the `nhgis*_codebook.txt` file downloaded alongside the data.

    Returns
    -------
    dict[str, str]
        maps each data column code (e.g. 'A38AA2020') to its age label
        (e.g. 'Under 1 year', '1 year', '85 years and over').
    '''
    import re

    text = Path(codebook_path).read_text()
    pattern = re.compile(r'^\s*(\w+):\s*\d{4}:\s*Persons:\s*(.+?)\s*$', re.MULTILINE)
    mapping = {code: label for code, label in pattern.findall(text)}
    if not mapping:
        raise ValueError(f'no age column mappings found in {codebook_path} codebook')
    return mapping


def _age_label_to_int(label: str) -> int:
    '''converts age label into an integer age

    Parameters
    ----------
    label : str
        an NHGIS age label, e.g. 'Under 1 year', '5 years', '85 years and over'.

    Returns
    -------
    int
        integer age, the top open-ended bucket maps to its start age.
    '''
    import re

    label = label.strip()
    if label.lower().startswith('under 1'):
        return 0
    match = re.match(r'(\d+)', label)
    if match:
        return int(match.group(1))
    raise ValueError(f'Could not parse an age from label: {label!r}')


def load_nhgis_age_population(data_path: Path, codebook_path: Path, state_filter: str = 'Texas') -> pd.DataFrame:
    '''loads and reshapes NHGIS single age year to long format

    Parameters
    ----------
    data_path : Path
        the `nhgis*_ts_nominal_state.csv` data file (wide: one row per
        state, one column per age code).
    codebook_path : Path
        the matching `nhgis*_codebook.txt`.
    state_filter : str
        value to match against the `STATE` column (default 'Texas').

    Returns
    -------
    pd.DataFrame
        Columns: `age` (int), `population` (int), long format, one row
        per single year of age (open-ended top bucket recoded to its
        starting age, e.g. '85 years and over' -> 85).
    '''
    code_to_label = parse_nhgis_codebook(codebook_path)
    raw = pd.read_csv(data_path)

    state_col = 'STATE' if 'STATE' in raw.columns else None
    if state_col is not None:
        raw = raw.loc[raw[state_col].str.contains(state_filter, case=False, na=False)]
    assert len(raw) == 1, (
        f'expected exactly one row for state_filter={state_filter!r}, got {len(raw)}'
    )

    age_cols = [c for c in raw.columns if c in code_to_label]
    if not age_cols:
        raise ValueError('no data columns matched the codebooks age codes')

    row = raw.iloc[0]
    records = [
        {'age': _age_label_to_int(code_to_label[col]), 'population': int(row[col])}
        for col in age_cols
    ]
    pop = pd.DataFrame(records).sort_values('age').reset_index(drop=True)
    return pop


def _population_in_range(age_population: pd.DataFrame, start: int, end: Optional[int]) -> float:
    '''sums population for ages in [start, end)

    Parameters
    ----------
    age_population : pd.DataFrame
        columns `age`, `population`.
    start : int
        inclusive lower age bound.
    end : int, optional
        exclusive upper age bound, `None` means unbounded (open-ended
        top bucket).

    Returns
    -------
    float
        summed population for ages in [start, end), or [start, inf) if
        `end` is `None`.
    '''
    mask = age_population['age'] >= start
    if end is not None:
        mask &= age_population['age'] < end
    return age_population.loc[mask, 'population'].sum()


def _overlap_population(age_population: pd.DataFrame, a_start: int, a_end: Optional[int], b_start: int, b_end: Optional[int]) -> float:
    '''computes the population in the intersection of two age ranges

    Parameters
    ----------
    age_population : pd.DataFrame
        columns `age`, `population`.
    a_start, a_end : int, optional
        first [start, end) age range, `a_end=None` is unbounded.
    b_start, b_end : int, optional
        second [start, end) age range, `b_end=None` is unbounded.

    Returns
    -------
    float
        population in the intersection of the two age ranges.
    '''
    lo = max(a_start, b_start)
    hi_candidates = [x for x in (a_end, b_end) if x is not None]
    hi = min(hi_candidates) if hi_candidates else None
    if hi is not None and lo >= hi:
        return 0.0
    return _population_in_range(age_population, lo, hi)


def build_rebinning_matrices(age_population: pd.DataFrame, original_band_edges: list[tuple[int, Optional[int]]], new_group_edges: list[tuple[int, Optional[int]]]) -> tuple[np.ndarray, np.ndarray]:
    '''builds the fraction and share weight matrices used to rebin the contact matrix
    frac(A, k) is the source side average or the fraction of new group A's population that falls into original band k
    share(B, l) is the is the destination-side split or the fraction of original band l's population that falls in new group B
    
    Parameters
    ----------
    age_population : pd.DataFrame
        output of `load_age_population`.
    original_band_edges : list of (start, end)
        original 5-year band edges, e.g. [(0, 5), (5, 10), ..., (75, None)].
    new_group_edges : list of (start, end)
        model's age group edges, e.g. [(0, 1), (1, 2), ..., (5, None)].

    Returns
    -------
    frac : np.ndarray, shape (n_new_groups, n_original_bands)
        frac[A, k] = pop(A ∩ k) / pop(A).
    share : np.ndarray, shape (n_original_bands, n_new_groups)
        share[l, B] = pop(B ∩ l) / pop(l).
    '''
    n_new = len(new_group_edges)
    n_orig = len(original_band_edges)

    frac = np.zeros((n_new, n_orig))
    for a_idx, (a_start, a_end) in enumerate(new_group_edges):
        pop_a = _population_in_range(age_population, a_start, a_end)
        for k_idx, (k_start, k_end) in enumerate(original_band_edges):
            overlap = _overlap_population(age_population, a_start, a_end, k_start, k_end)
            frac[a_idx, k_idx] = overlap / pop_a if pop_a > 0 else 0.0

    share = np.zeros((n_orig, n_new))
    for l_idx, (l_start, l_end) in enumerate(original_band_edges):
        pop_l = _population_in_range(age_population, l_start, l_end)
        for b_idx, (b_start, b_end) in enumerate(new_group_edges):
            overlap = _overlap_population(age_population, b_start, b_end, l_start, l_end)
            share[l_idx, b_idx] = overlap / pop_l if pop_l > 0 else 0.0

    return frac, share


def rebin_contact_matrix(raw_matrix: pd.DataFrame, age_population: pd.DataFrame, original_band_edges: list[tuple[int, Optional[int]]], new_group_edges: Optional[list[tuple[int, Optional[int]]]] = None, new_group_labels: Optional[list[str]] = None) -> pd.DataFrame:
    '''rebins the original contact matrix to the desired age structure using C_new(A, B) = sum_k sum_l frac(A, k) * M(k, l) *
    share(B, l) with vectorized operations
    
    Parameters
    ----------
    raw_matrix : pd.DataFrame
        output of `load_raw_contact_matrix` (n_orig x n_orig).
    age_population : pd.DataFrame
        output of `load_age_population`.
    original_band_edges : list of (start, end)
        must be in the same order as `raw_matrix`'s rows/columns.
    new_group_edges : list of (start, end), optional
        defaults to `config['age_structure']['age_group_edges']` paired consecutively.
    new_group_labels : list of str, optional
        defaults to `config['age_structure']['age_group_labels'].

    Returns
    -------
    pd.DataFrame
        n_new x n_new contact matrix indexed/columned by new group label.
    '''
    new_group_labels = new_group_labels or config['age_structure']['age_group_labels']
    if new_group_edges is None:
        edges = config['age_structure']['age_group_edges']
        new_group_edges = list(zip(edges[:-1], edges[1:]))

    frac, share = build_rebinning_matrices(
        age_population, original_band_edges, new_group_edges
    )
    M = raw_matrix.to_numpy()
    C_new = frac @ M @ share  # (n_new, n_orig) @ (n_orig, n_orig) @ (n_orig, n_new)

    return pd.DataFrame(C_new, index=new_group_labels, columns=new_group_labels)


def normalize_to_unit_spectral_radius(matrix: pd.DataFrame) -> pd.DataFrame:
    '''nomralizes the new contact matrix to unit spectral radius for convenient R0 estimate
    Parameters
    ----------
    matrix : pd.DataFrame
        A square contact matrix (raw or rebinned).

    Returns
    -------
    pd.DataFrame
        same shape/index/columns, rescaled by 1/rho(C). Symmetric input
        stays symmetric (this is a uniform scalar rescale, not a
        row/column-specific one).

    Raises
    ------
    valueError
        if the matrix's dominant eigenvalue is non-positive.
    '''
    rho = np.linalg.eigvals(matrix.to_numpy()).real.max()
    if rho <= 0:
        raise ValueError(f'contact matrix has non-positive dominant eigenvalue ({rho})')
    return matrix / rho


def validate_contact_matrix(matrix: pd.DataFrame, expected_dim: int = config['age_structure']['n_age_groups']) -> None:
    '''checks matrix dimensions, non negativity, and no nans in contact matrix
    Parameters
    ----------
    matrix : pd.DataFrame
        A square contact matrix (raw or rebinned).
    expected_dim : int
        expected number of rows/columns, by default config['age_structure']['n_age_groups'].

    Returns
    -------
    none

    Raises
    ------
    assertionError
        if any check fails.
    '''
    assert matrix.shape == (expected_dim, expected_dim), (
        f'Expected a {expected_dim}x{expected_dim} matrix, got {matrix.shape}'
    )
    assert (matrix.to_numpy() >= 0).all(), 'Contact matrix contains negative entries'
    assert not matrix.isna().any().any(), 'Contact matrix contains unexpected NaNs'


def build_processed_contact_matrix(cfg=config) -> None:
    '''runs the full contact matrix pipeline and saves to data/processed/
    Parameters
    ----------
    cfg : module
        config module (or object exposing the same attributes).

    Returns
    -------
    none
    '''
    original_edges = [
        (start, start + cfg['contact_matrix']['original_contact_band_width'])
        for start in range(0, 75, cfg['contact_matrix']['original_contact_band_width'])
    ] + [(75, None)]

    raw = load_raw_contact_matrix(cfg['contact_matrix']['raw_contact_matrix_path'], cfg['contact_matrix']['original_contact_band_labels'])
    validate_contact_matrix(raw, expected_dim=len(original_edges))

    age_pop = load_nhgis_age_population(cfg['data']['raw_nhgis_data_path'], cfg['data']['raw_nhgis_codebook_path'])
    rebinned = rebin_contact_matrix(raw, age_pop, original_edges)
    validate_contact_matrix(rebinned, expected_dim=cfg['age_structure']['n_age_groups'])
    rebinned = normalize_to_unit_spectral_radius(rebinned)
    validate_contact_matrix(rebinned, expected_dim=cfg['age_structure']['n_age_groups'])

    Path(cfg['paths']['data_processed_dir']).mkdir(parents=True, exist_ok=True)
    rebinned.to_csv(cfg['paths']['processed_contact_matrix_path'])


def load_processed_contact_matrix(cfg=config) -> pd.DataFrame:
    '''loads the rebinned contact matrix
    Parameters
    ----------
    cfg : module
        config module (or object exposing the same attributes).

    Returns
    -------
    pd.DataFrame
        the rebinned, normalized contact matrix, indexed/columned by age
        group label.
    '''
    return pd.read_csv(cfg['paths']['processed_contact_matrix_path'], index_col=0)


def population_by_age_group(age_population: pd.DataFrame, group_edges: Optional[list[tuple[int, Optional[int]]]] = None) -> np.ndarray:
    '''sums single year of age population counts into the desired age groups
    Parameters
    ----------
    age_population : pd.DataFrame
        output of `load_nhgis_age_population` (or `load_raw_contact_matrix`'s
        population input more generally): columns `age`, `population`.
    group_edges : list of (start, end), optional
        defaults to `config['age_structure']['age_group_edges']` paired consecutively.

    Returns
    -------
    np.ndarray, shape (config['age_structure']['n_age_groups'],)
        population per model age group, in `config['age_structure']['age_group_labels']` order.
    '''
    if group_edges is None:
        edges = config['age_structure']['age_group_edges']
        group_edges = list(zip(edges[:-1], edges[1:]))
    return np.array(
        [_population_in_range(age_population, start, end) for start, end in group_edges],
        dtype=float,
    )