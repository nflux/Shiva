import torch

def roll(tensor: torch.tensor, rollover: int) -> torch.tensor:
    """
    Roll over the first axis of a tensor

    Args:
        tensor: tensor to be rolled over
        rollover: index the roll over occurs

    Returns:
        torch.tensor: rolled over tensor
    """
    return torch.cat((tensor[-rollover:], tensor[:-rollover]))

def roll2(tensor: torch.tensor, rollover: int) -> torch.tensor:
    """
    Roll over the second axis of a tensor

    Args:
        tensor: tensor to be rolled over
        rollover: index the roll over occurs

    Returns:
        torch.tensor: rolled over tensor
    """
    return torch.cat((tensor[:,-rollover:], tensor[:,:-rollover]), dim=1)