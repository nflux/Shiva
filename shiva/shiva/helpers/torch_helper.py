import torch

def normalize_rows(t: torch.tensor):
    sums = t.sum(dim=-1)
    return t / sums.reshape(-1, 1)

def normalize_branches(t: torch.tensor, action_space: tuple, f=None):
    """

    Expected t.shape to be of [batch_size, flattened actions]"""
    device = t.device
    f = normalize_rows if f is None else f
    # t2 = t
    acc = 0
    for size in action_space:
        ixs_range = torch.arange(acc, acc+size).to(device)
        vals = t.index_select(-1, ixs_range).clone().abs()
        t[..., ixs_range] = f(vals)
        acc += size
    return t