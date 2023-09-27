"""
readout



@Author: linlin
@Date: 25.05.23
"""


def get_readout(key):
    """
    Return the readout function.
    """
    if key == 'sum':
        from torch_geometric.nn.aggr import SumAggregation
        return SumAggregation()
    elif key == 'mean':
        from torch_geometric.nn.aggr import MeanAggregation
        return MeanAggregation()
    elif key == 'max':
        from torch_geometric.nn.aggr import MaxAggregation
        return MaxAggregation()
    elif key == 'min':
        from torch_geometric.nn.aggr import MinAggregation
        return MinAggregation()
    elif key == 'meanmax':
        raise NotImplementedError(
            'MeanMaxAggregation is not implemented in PyG.'
        )
    elif key == 'set2set':
        from torch_geometric.nn.aggr import Set2Set
        return Set2Set()
    elif key == 'transformer':
        raise NotImplementedError(
            'TransformerAggregation is not implemented in PyG.'
        )
    elif key is None:
        return None
    else:
        raise ValueError('Invalid readout key. Please check again.')
