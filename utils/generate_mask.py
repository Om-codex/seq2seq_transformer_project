import torch
def generate_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)) # torch.tril -> lower triangle would be one and rest would be 0
    return mask