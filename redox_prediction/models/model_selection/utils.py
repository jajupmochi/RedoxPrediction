"""
utils



@Author: linlin
@Date: 18.08.23
"""
import numpy as np

from typing import List, Union, Iterable

import networkx


def get_params_grid_task(
        metric: str,
        model_type: str,
):
    # if the precomputed matrix is a Gram matrix:
    if metric == 'dot-product':
        if model_type == 'reg':
            return {'alpha': np.logspace(-10, 10, num=21, base=10)}
            # from sklearn.kernel_ridge import KernelRidge
            # from redox_prediction.utils.distances import rmse
            # estimator = KernelRidge(kernel='precomputed')
            # # scoring = 'neg_root_mean_squared_error'
            # perf_eval = rmse
        elif model_type == 'classif':
            return {'C': np.logspace(-10, 10, num=21, base=10)}
            # from sklearn.svm import SVC
            # from redox_prediction.utils.distances import accuracy
            # estimator = SVC(kernel='precomputed')
            # # scoring = 'accuracy'
            # perf_eval = accuracy
        else:
            raise ValueError('"model_type" must be either "reg" or "classif".')
    # if the precomputed matrix is a distance matrix:
    elif metric == 'distance':
        return {'n_neighbors': [3, 5, 7, 9, 11]}
    else:
        raise ValueError('"metric" must be either "dot-product" or "distance".')


def get_submatrix_by_index(
        matrix: np.ndarray,
        index: Union[List[networkx.Graph], Iterable],
        index_row: Union[List[networkx.Graph], Iterable] = None,
        idx_key: str = 'id',
):
    """
    Get a submatrix from a matrix by index.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix from which to extract the submatrix.

    index : list of networkx.Graph or iterable
        The index of the submatrix. If `index_row` is provided, `index` will
        be used as the index of the columns of the submatrix. Otherwise, it
        will be used as the index of both the rows and the columns.
        If a list of graphs is provided, the submatrix will be extracted from
        the matrix using the graph id.

    index_row : list of networkx.Graph or iterable
        The index of the rows of the submatrix.
        If a list of graphs is provided, the submatrix will be extracted from
        the matrix using the graph id. The type of `index_row` must be the same
        as the type of `index`.

    idx_key : str
        The key of the graph id in the graph attributes.

    Returns
    -------
    np.ndarray
        The submatrix.
    """
    # Check if the types of `index` and `index_row` are the same:
    if index_row is not None:
        if type(index) != type(index_row):
            raise TypeError(
                '"index" and "index_row" must be of the same type.'
            )

    # If `index` is a list of graphs, extract the graph ids:
    if isinstance(index, list) and isinstance(index[0], networkx.Graph):
        index = [int(graph.graph[idx_key]) for graph in index]
        if index_row is not None:
            index_row = [int(graph.graph[idx_key]) for graph in index_row]

    # Extract the submatrix:
    if index_row is not None:
        return matrix[np.ix_(index_row, index)]
    else:
        return matrix[np.ix_(index, index)]


def get_y_scaler(
        y: np.ndarray,
        y_scaling: str = 'std',
):
    """
    Get the scaler for the targets.

    Parameters
    ----------
    y : np.ndarray
        The targets.

    y_scaling : str, optional (default='std')
        The type of scaler to use. It can be one of the following:
        - 'std': StandardScaler
        - 'minmax': MinMaxScaler
        - 'none': No scaler

    Returns
    -------
    sklearn.preprocessing.StandardScaler or sklearn.preprocessing.MinMaxScaler or None
        The scaler for the targets.
    """
    if y_scaling == 'std':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif y_scaling == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif y_scaling == 'none':
        scaler = None
    else:
        raise ValueError(
            '"y_scaling" must be one of the following: '
            '"std", "minmax", "none".'
        )

    if scaler is not None:
        scaler.fit(y.reshape(-1, 1))

    return scaler
