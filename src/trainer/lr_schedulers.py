import torch


def get_scheduler(scheduler: str = 'reduce', optimizer = None, **kwargs):
    if scheduler is None:
        return None

    scheduler = scheduler.lower()

    if scheduler.startswith('const'):
        from torch.optim.lr_scheduler import ConstantLR as LrScheduler

    elif scheduler.startswith('cosine'):
        if 'warm_restart' in scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as LrScheduler
            if 'T_0' not in kwargs.items():
                kwargs['T_0'] = 30
            if 'T_mult ' not in kwargs.items():
                kwargs['T_mult '] = 20
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR as LrScheduler
            if 'T_max' not in kwargs.items():
                kwargs['T_max'] = 70
        if 'eta_min' not in kwargs.items():
            kwargs['eta_min'] = 1e-7
    
    elif scheduler == 'multiplicative':
        from torch.optim.lr_scheduler import MultiplicativeLR as LrScheduler
        if 'lr_lambda' not in kwargs.items():
            kwargs['lr_lambda'] = lambda epoch: 0.95
    
    elif scheduler == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR as LrScheduler
        if 'milestones' not in kwargs.items():
            kwargs['milestones'] = [100, 250, 500]
    
    elif scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR as LrScheduler
        if 'step_size' not in kwargs.items():
            kwargs['step_size'] = 100
    
    elif scheduler == 'linear':
        from torch.optim.lr_scheduler import LinearLR as LrScheduler
    
    elif scheduler == 'exponential':
        from torch.optim.lr_scheduler import ExponentialLR as LrScheduler
        if 'gamma' not in kwargs.items():
            kwargs['gamma'] = 0.369
    
    # elif scheduler == 'chained':
    #     from torch.optim.lr_scheduler import ChainedScheduler as LrScheduler
    
    # elif scheduler == 'sequential':
    #     from torch.optim.lr_scheduler import SequentialLR as LrScheduler
    
    # elif scheduler == 'single_cycle':
    #     from torch.optim.lr_scheduler import OneCycleLR as LrScheduler
    
    elif scheduler == 'cyclic':
        from torch.optim.lr_scheduler import CyclicLR as LrScheduler
        if 'base_lr' not in kwargs.items():
            kwargs['base_lr'] = 0.01
        if 'max_lr' not in kwargs.items():
            kwargs['max_lr'] = 10 * kwargs['base_lr']
    
    elif scheduler == 'reduce':
        from torch.optim.lr_scheduler import ReduceLROnPlateau as LrScheduler
        if 'patience' not in kwargs.items():
            kwargs['patience'] = 3
        if 'factor' not in kwargs.items():
            kwargs['factor'] = 0.369
    else:
        return None

    if 'verbose' not in kwargs.items():
        kwargs['verbose'] = True

    if isinstance(optimizer, torch.optim.Optimizer):
        return LrScheduler(optimizer = optimizer, **kwargs)
    return LrScheduler
