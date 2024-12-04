# The code in this file is adapted from https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers,num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(5600, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(
              num_hiddens * 2, num_hiddens * 2))

        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2 * num_hiddens, num_classes)
        nn.init.xavier_uniform_(self.w_omega, gain=1)
        nn.init.xavier_uniform_(self.u_omega, gain=1)

    def forward(self, x, return_feature=False):
        x = x.to(torch.long)
        embed = self.embedding(x)
        out, _ = self.encoder(embed)

        u = torch.tanh(torch.matmul(out, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = out * att_score
        feat = torch.sum(scored_x, dim=1)
        out = self.decoder(feat)
        if (return_feature):
            return feat, out
        else:
            return out