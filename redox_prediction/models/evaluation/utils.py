"""
utils



@Author: linlin
@Date: 18.08.23
"""


def get_estimator(
        metric: str,
        model_type: str,
):
    # if the precomputed matrix is a Gram matrix:
    if metric == 'dot-product':
        if model_type == 'reg':
            from sklearn.kernel_ridge import KernelRidge
            from redox_prediction.utils.distances import rmse
            estimator = KernelRidge(kernel='precomputed')
            # scoring = 'neg_root_mean_squared_error'
            perf_eval = rmse
        elif model_type == 'classif':
            from sklearn.svm import SVC
            from redox_prediction.utils.distances import accuracy
            estimator = SVC(kernel='precomputed')
            # scoring = 'accuracy'
            perf_eval = accuracy
        else:
            raise ValueError('"model_type" must be either "reg" or "classif".')
    # if the precomputed matrix is a distance matrix:
    elif metric == 'distance':
        if model_type == 'reg':
            from sklearn.neighbors import KNeighborsRegressor
            from redox_prediction.utils.distances import rmse
            estimator = KNeighborsRegressor(metric='precomputed')
            # scoring = 'neg_root_mean_squared_error'
            perf_eval = rmse
        elif model_type == 'classif':
            from sklearn.neighbors import KNeighborsClassifier
            from redox_prediction.utils.distances import accuracy
            estimator = KNeighborsClassifier(metric='precomputed')
            # scoring = 'accuracy'
            perf_eval = accuracy
        else:
            raise ValueError('"model_type" must be either "reg" or "classif".')
    else:
        raise ValueError('"metric" must be either "dot-product" or "distance".')

    return estimator, perf_eval
