import torch

def roll(tensor, rollover):
    '''
    Roll over the first axis of a tensor
    '''
    return torch.cat((tensor[-rollover:], tensor[:-rollover]))

def roll2(tensor, rollover):
    '''
    Roll over the second axis of a tensor
    '''
    return torch.cat((tensor[:,-rollover:], tensor[:,:-rollover]), dim=1)