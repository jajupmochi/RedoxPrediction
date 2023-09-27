"""
early_stopping



@Author: linlin
@Date: 03.08.23
"""


def get_early_stopping_metric(
        early_stopping_metric: str,
        mode: str = 'min'
) -> callable:
    """
    Returns a function to compute the early stopping metric.

    Parameters
    ----------
    early_stopping_metric: str
        The metric to use for early stopping.
    mode: str, optional
        The model_type of the metric. Either 'min' or 'max'.

    Returns
    -------
    function
        The function to compute the early stopping metric.
    """
    if early_stopping_metric == 'loss':
        if mode == 'min':
            return lambda x, y: x < y
        elif mode == 'max':
            return lambda x, y: x > y
        else:
            raise ValueError(
                f'Invalid model_type {mode}. '
                f'Expected either "min" or "max".'
            )
    else:
        raise ValueError(
            f'Invalid metric {early_stopping_metric}. '
            f'Expected either "loss".'
        )
