def perturb_optimizer(optimizer, new_values: dict):
    for param_group in optimizer.param_groups:
        for key, val in new_values.items():
            param_group[key] = val
    return optimizer