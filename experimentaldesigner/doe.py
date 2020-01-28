# -*- coding: utf-8 -*-
import pandas as pd
from . import app
import numpy as np
import dash_html_components as html
from pyDOE import lhs
from diversipy.hycusampling import transform_spread_out, sukharev_grid, maximin_reconstruction, random_k_means, random_uniform
from diversipy.indicator import unanchored_L2_discrepancy, mean_dist_to_boundary, separation_dist
from scipy.stats.distributions import norm, poisson, beta
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .common import cell_format

distribution_map = {
    'norm': norm,
    'poisson': poisson,
    'beta': beta
}


def _map_random_matrix_to_df(random_matrix, sampling_df):

    experiments = []

    app.logger.info("random matrix: {}".format(random_matrix))

    def remap(value, factorrange):
        scale = np.abs(factorrange[1]-factorrange[0])
        return factorrange[0]+(value*scale)
    for experiment_ind in range(random_matrix.shape[0]):
        experiment = {}
        for factor in range(random_matrix.shape[1]):
            row = sampling_df.iloc[factor, :]
            value = remap(random_matrix[experiment_ind][factor], row['range'])
            experiment[row['name']] = value
        experiments.append(experiment)

    return pd.DataFrame(experiments)


def _get_sampling_df(feature_names: list, lower_lim: list, upper_lim: list, num_samples: int) -> pd.DataFrame:
    """Convert the user inputs into a dataframe, also have the weights normalized into samples along each dimension. 

    Arguments:
        feature_names {list} -- list of strings of feature names
        lower_lim {list} --list of floats of lower limits 
        upper_lim {list} -- list of floats of upper limits 
        num_samples {int} -- [description]

    Returns:
        pd.DataFrame -- [description]
    """
    sampling_dicts = []

    # assert len(feature_names) == len(
    #     weights) == len(lower_lim) == len(upper_lim)

    # importances = _get_num_samples(weights, num_samples)
    for name, lower, upper in zip(feature_names, lower_lim, upper_lim):
        sampling_dict = {}
        if lower >= upper:
            app.logger.warning(
                'lower limit is greater than upper limit, will flip them')
            lower_ = lower
            upper_ = upper
            upper = lower_
            lower = upper_

        sampling_dict['name'] = name
        sampling_dict['range'] = (lower, upper)

        sampling_dicts.append(sampling_dict)
    df = pd.DataFrame(sampling_dicts)
    app.logger.info('The sampling dicts looks like {}'.format(df))
    app.logger.info('The length of the df is {}'.format(len(df)))
    return df


def _get_num_samples(importances: list, num_samples: int) -> list:
    """Convert importances to number of samples along each dimension.

    Arguments:
        importances {list} -- list of importances from 
        num_samples {int} -- total number of samples given by the users

    Returns:
        list -- [description]
    """
    normalized_importances /= normalized_importanes.sum()
    samples_for_factor = [num_samples %
                          importance for importance in normalized_importances]

    samples = samples_for_factor.sum()
    if samples != num_samples:
        app.logger.warning('The requested number of samples is {}, the actual one is {}'.format(
            num_samples, samples))
        app.logger.warning(
            'For this reason, I will distribute the remaining samples to the most undersampled classes')

    return samples_for_factor


def _apply_dist(grid: np.array, distribution: str):
    if distribution not in list(distribution_map.keys()):
        app.logger.info('Applying no distribution to sampling')
        distribution = None
    elif distribution == 'norm':
        grid = norm(loc=0, scale=1).ppf(grid)
    elif distribution == 'poisson':
        grid = poisson().ppf(grid)
    elif distribution == 'beta':
        grid = beta().ppf(grid)

    return grid


def build_lhs_grid(factor_ranges: pd.DataFrame, num_samples: int = None, criterion: str = 'centermaximin', distribution: str = None, spacefilling: bool = False) -> pd.DataFrame:
    lhd = lhs(len(factor_ranges), samples=num_samples, criterion=criterion)
    app.logger.info('latin hypercube design {}'.format(lhd))
    # ToDo: make this more elegant
    _apply_dist(lhd, distribution)

    # spread out if needed
    if spacefilling:
        lhd = transform_spread_out(lhd)

    app.logger.info('latin hypercube design 2 {}'.format(lhd))

    df = _map_random_matrix_to_df(lhd, factor_ranges)
    app.logger.info('latin hypercube design 3 {}'.format(lhd))
    return df


def build_sukharev_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = sukharev_grid(num_samples, len(factor_ranges))
    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df


def build_maxmin_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = maximin_reconstruction(num_samples, len(factor_ranges))
    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df


def build_random_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = random_uniform(num_samples, len(factor_ranges))

    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df


def build_kmeans_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = random_k_means(num_samples, len(factor_ranges))

    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df


def measure_sampling_quality(sampled_frame):
    in_unitcube = MinMaxScaler(feature_range=(
        0.001, 0.999)).fit_transform(sampled_frame.values)
    boundary_dist = mean_dist_to_boundary(in_unitcube)
    l2 = unanchored_L2_discrepancy(in_unitcube)
    min_dist = separation_dist(in_unitcube)

    table = html.Table(
        # Header
        [html.Tr([html.Th('indicator'), html.Th('value')])] +

        # Body
        [
            html.Tr([
                html.Td('l2 discrepancy (should be minimized)'), html.Td(cell_format(l2))]),
            html.Tr([html.Td('minimum pairwise distance (should be maximized)'), html.Td(
                cell_format(min_dist))]),
            html.Tr([html.Td('mean distance to boundary'),
                     html.Td(cell_format(l2))])
        ]

    )

    return table


def run_mds(grid):
    projected = MDS(2).fit_transform(StandardScaler().fit_transform(grid))
    return projected
