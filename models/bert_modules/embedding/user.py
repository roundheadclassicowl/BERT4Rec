import torch.nn as nn
import torch
import math


class UserEmbedding(nn.Module):

    def __init__(self, user_size, embed_size):
        super().__init__()
        # super().__init__(user_size, embed_size, padding_idx=0)
        
        # each vector is associated with user_id
        # Compute the user encodings once in log space.
        self.ue = nn.Embedding(user_size, embed_size) # user_size tensors with shape d_model

    def forward(self, x):
        # print("embedding/user------x shape", self.ue.weight.shape)
        T = 100
        rtv = self.ue.weight[x]
        rtv = rtv.unsqueeze(1).repeat(1, T, 1) # .repeat(batch_size, T, 1) 128*100*256
        # print("embedding/user------forward shape", rtv.shape)
        return rtv
