from .distance import *
from .regression import *
from .classification import *


def get_loss_fn(loss_fn: str = None, task: str = None):
    assert (loss_fn is not None) or (task is not None)

    if loss_fn is None:
        if task == 'classification':
            loss_fn = 'cross-entropy'
        elif task == 'regression':
            loss_fn = 'mse'
    loss_fn = loss_fn.lower()  # Normalize the string to lowercase

    # Regression losses
    if loss_fn in ['mse','l2']:
        return MSELoss()

    elif loss_fn in ['mae','l1']:
        return MAELoss()

    elif loss_fn in ['huber']:
        return HuberLoss()

    elif loss_fn in ['log-cosh']:
        return LogCoshLoss()

    elif loss_fn in ['mape']:
        return MAPELoss()

    elif loss_fn in ['r-squared']:
        return RSquaredLoss()

    elif loss_fn in ['r-squared-adjusted']:
        return AdjustedRSquaredLoss()

    # Classification losses
    elif loss_fn in ['nll','negative-log-likelihood']:
        return NLLLoss()

    elif loss_fn in ['cross-entropy']:
        return CrossEntropyLoss()

    elif loss_fn in ['binary-cross-entropy','bce']:
        return BCELoss()

    elif loss_fn in ['focal']:
        return FocalLoss()

    raise ValueError(f"Unknown loss_fn: {loss_fn}. Please choose a valid loss_fn.")


