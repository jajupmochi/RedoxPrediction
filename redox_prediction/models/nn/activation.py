"""
activation



@Author: linlin
@Date: 25.05.23
"""

def get_activation(key):
    """
    Return the activation function.
    """
    if key == 'relu':
        from torch.nn import ReLU
        return ReLU()
    elif key == 'silu':
        from torch.nn import SiLU
        return SiLU()
    elif key == 'sigmoid':
        from torch.nn import Sigmoid
        return Sigmoid()
    elif key == 'softmax':
        from torch.nn import Softmax
        return Softmax(dim=1)
    elif key == 'log_softmax':
        from torch.nn import LogSoftmax
        return LogSoftmax(dim=1)  # @TODO: dim=1?
    elif key == 'tanh':
        from torch.nn import Tanh
        return Tanh()
    elif key is None:
        return None
    else:
        raise NotImplementedError(
            'Activation function {} is not implemented.'.format(key)
        )
