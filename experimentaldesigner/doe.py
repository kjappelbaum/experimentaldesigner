# -*- coding: utf-8 -*-
import pandas as pd
from . import app
import numpy as np
from pyDOE import lhs
from diversipy.hycusampling import transform_spread_out, sukharev_grid, maximin_reconstruction
from scipy.stats.distributions import norm, poisson, beta

distribution_map = {
    'norm': norm,
    'poisson': poisson,
    'beta': beta
}


def _map_random_matrix_to_df(random_matrix, sampling_df):

    experiments = []

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

    return pd.DataFrame(sampling_dicts)


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


def build_lhs_grid(factor_ranges: pd.DataFrame, num_samples: int = None, criterion: str = None, distribution: str = None, spacefilling: bool = True) -> pd.DataFrame:
    lhd = lhs(len(factor_ranges), samples=num_samples, criterion=criterion)

    # ToDO: make this more elegant
    if distribution not in list(distribution_map.keys()):
        distribution = None
    elif distribution == 'norm':
        lhd = norm(loc=0, scale=1).ppf(lhd)
    elif distribution == 'poisson':
        lhd = poisson().ppf(lhd)
    elif distribution == 'beta':
        lhd = beta().ppf(lhd)

    # subsample if the weights are not equal

    # spread out if needed
    if spacefilling:
        lhd = transform_spread_out(lhd)

    df = _map_random_matrix_to_df(lhd, factor_ranges)

    return df


def build_sukharev_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = sukharev_grid(num_points, len(factor_ranges))
    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df


def build_maxmin_grid(factor_ranges: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    grid = maximin_reconstruction(num_points, dimension)
    df = _map_random_matrix_to_df(grid, factor_ranges)

    return df
