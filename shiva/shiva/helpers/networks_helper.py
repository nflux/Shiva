def mod_optimizer(optimizer: object, new_values: dict) -> object:
    '''
    Modified the given optimizer using the new values

    Args:
        optimizer: optimizer to be modified
        new_values: dictionary whose keys are each `param_groups` keys and values are the new values to inject into the optimizer

    Returns:
        Optimizer

    '''
    for param_group in optimizer.param_groups:
        for key, val in new_values.items():
            param_group[key] = val
    return optimizer
