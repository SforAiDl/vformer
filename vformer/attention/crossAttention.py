from typing import ForwardRef

import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        if not in_dim == out_dim:
            self.l1 = nn.Linear(in_dim, out_dim)
        else:
            self.l1 = nn.Identity()

    def forward(self, x):
        return self.l1(x)


class CrossAttention(nn.Module):
    def __init__(self, cls_dim, patch_dim, inner_dim):
        super().__init__()
        self.fl = Projection(cls_dim, patch_dim)
        self.gl = Projection(patch_dim, cls_dim)
        self.to_k = nn.Linear(patch_dim, inner_dim)
        self.to_v = nn.Linear(patch_dim, patch_dim)
        self.to_q = nn.Linear(patch_dim, inner_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, cls, patches):
        # cls (batch,data,1,embedding) patch (batch,data,patch,embedding)
        cls = self.fl(cls)
        x = torch.cat([cls, patches], dim=2)
        q = self.to_q(cls)  # (batch,data,1,inner_embedding)
        k = self.to_k(x)  # (batch,data,patch+1,inner_embedding)
        v = self.to_v(x)  # (batch,data,patch+1,patch_dim)
        k = torch.transpose(k, 2, 3)  # (batch,data,inner_embedding,patch+1)
        attention = v @ q  # (batch,data,patch+1,1)
        attention = self.softmax(attention)
        attention_value = v * attention  # (batch,data,patch+1,patch_dim)
        ycls = cls + attention_value
        return ycls
