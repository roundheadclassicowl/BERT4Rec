import torch
import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .user import UserEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, user_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.user = UserEmbedding(user_size=user_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, user_idx):
        # user_idx = torch.LongTensor([user_idx])
        # print("embedding/bert------user indices shape", user_idx.unsqueeze(1).repeat(1, 100).shape)
        x = self.token(sequence)
        # print("embedding/bert------token embed shape", x.shape)
        pe = self.position(sequence)
        # print("embedding/bert------position embed shape", pe.shape)
        x += pe
        ue = self.user(user_idx)
        # print("embedding/bert------user embed shape", ue.shape) # need to be B * T * V
        x += ue # + self.segment(segment_label)
        return self.dropout(x)
