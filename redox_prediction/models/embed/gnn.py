"""
gnn



@Author: linlin
@Date: 27.05.23
"""
import numpy as np


def _compute_D_gnn(
        G_app, y_app, G_test, y_test, estimator, param_grid,
        mode, fit_test, y_distance=None, **kwargs
):
    # Compute Gram matrix:
    from redox_prediction.models.model_selection.pretrain_gnn import model_selection_for_gnn
    model, perf_app, perf_test, embed_app, embed_test, params = model_selection_for_gnn(
        G_app, y_app, G_test, y_test,
        estimator,
        param_grid,
        mode,
        fit_test=fit_test,
        parallel=False,
        # n_jobs=1,
        read_gm_from_file=False,
        verbose=True,  # @TODO # (True if len(G_app) > 1000 else False),
        **kwargs,
    )
    # Compute distances between elements in embedded space:
    if not fit_test:
        dis_mat = np.zeros((len(G_app), len(G_app)))
        for i in range(len(G_app)):
            for j in range(i + 1, len(G_app)):
                dis_mat[i, j] = y_distance(embed_app[i], embed_app[j])
                dis_mat[j, i] = dis_mat[i, j]

        return dis_mat
    else:
        return perf_app, perf_test, embed_app, embed_test, model


def compute_D_gcn(
        G_app, y_app, G_test, y_test,
        y_distance=None,
        mode='reg', unlabeled=False, ed_method='bipartite',
        descriptor='atom_bond_types',
        fit_test=False,
        **kwargs
):
    """
    Return the distance matrix computed by GCN.
    """
    # @todo ['softmax', 'log_softmax'],
    clf_activation = ('sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')
    # Get parameter grid:
    param_grid = {
        'hidden_feats': [32, 64],
        'message_steps': [2, 3],
        'agg_activation': ['relu'],
        'readout': ['mean'],
        'predictor_hidden_feats': [32, 64],
        'predictor_n_hidden_layers': [1],
        'predictor_activation': ['relu'],
        'predictor_clf_activation': [clf_activation],
        'batch_size': [32],
    }

    from redox_prediction.models.nn.gcn import GCN
    estimator = GCN
    return _compute_D_gnn(
            G_app, y_app, G_test, y_test, estimator, param_grid,
            mode, fit_test, y_distance=y_distance, **kwargs
    )


def compute_D_gat(
        G_app, y_app, G_test, y_test,
        y_distance=None,
        mode='reg', unlabeled=False, ed_method='bipartite',
        descriptor='atom_bond_types',
        fit_test=False,
        **kwargs
):
    """
    Return the distance matrix computed by GAT.
    """
    # @todo ['softmax', 'log_softmax'],
    clf_activation = ('sigmoid' if kwargs.get('n_classes') == 2 else 'log_softmax')
    # Get parameter grid:
    param_grid = {
        'hidden_feats': [32, 64],
        'n_heads': [4],  # [4, 8],
        'concat_heads': [True],  # [True, False],
        'message_steps': [2, 3],
        'attention_drop': [0.],
        'agg_activation': ['relu'],
        'readout': ['mean'],
        'predictor_hidden_feats': [32, 64],
        'predictor_n_hidden_layers': [1],
        'predictor_activation': ['relu'],
        'predictor_clf_activation': [clf_activation],
        'batch_size': [32],
    }

    from redox_prediction.models.nn.gat import GAT
    estimator = GAT
    return _compute_D_gnn(
            G_app, y_app, G_test, y_test, estimator, param_grid,
            mode, fit_test, y_distance=y_distance, **kwargs
    )