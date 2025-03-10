# coding: utf-8

import torch

def cosine_distance(embeddings1, embeddings2):
    """ Calculate cosine similarities between embeddings """

    # Mean across time dimension gets single vector per embedding
    mu_embed1 = torch.mean(torch.from_numpy(embeddings1), dim=0)
    mu_embed2= torch.mean(torch.from_numpy(embeddings2), dim=0)
    
    # Cosine similarity between mean vectors
    codist_embeds = torch.nn.functional.cosine_similarity(
        mu_embed1.unsqueeze(0), 
        mu_embed2.unsqueeze(0)
    ).item()
    
    return codist_embeds