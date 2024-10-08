import torch.optim as optim


def get_optimizer(optimizer_name):
    """
    Returns a PyTorch optimizer based on the input string.
    
    Parameters:
        - optimizer_name (str): The name of the optimizer as a string (e.g., 'SGD', 'Adam').
        - model_params: The parameters of the model (model.parameters()).
        - lr (float): The learning rate for the optimizer.
        - kwargs: Additional keyword arguments for specific optimizers.
    
    Returns:
        - optimizer: A PyTorch optimizer object.
    """
    optimizer_name = optimizer_name.lower()  # Normalize the string to lowercase

    if optimizer_name == 'sgd':
        from torch.optim import SGD as Optimizer
    elif optimizer_name == 'asgd':
        from torch.optim import ASGD as Optimizer
    elif optimizer_name == 'adam':
        from torch.optim import Adam as Optimizer
    elif optimizer_name == 'adamw':
        from torch.optim import AdamW as Optimizer
    elif optimizer_name == 'adamax':
        from torch.optim import Adamax as Optimizer
    elif optimizer_name == 'adadelta':
        from torch.optim import Adadelta as Optimizer
    elif optimizer_name == 'adagrad':
        from torch.optim import Adagrad as Optimizer
    elif optimizer_name == 'rmsprop':
        from torch.optim import RMSprop as Optimizer
    elif optimizer_name == 'lbfgs':
        from torch.optim import LBFGS as Optimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Please choose a valid optimizer.")

    return Optimizer
