import torch
import torch.nn as nn


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cls, patches):
        # cls (data,1,cls_dim) patch (data,patch,patch_dim)
        cls = self.fl(cls)
        x = torch.cat([cls, patches], dim=-2)
        q = self.to_q(cls)  # (data,1,inner_embedding)
        k = self.to_k(x)  # (data,patch+1,inner_embedding)
        v = self.to_v(x)  # (data,patch+1,patch_dim)
        k = torch.transpose(k, -2, -1)  # (data,inner_embedding,patch+1)
        attention = q @ k  # (data,1,patch+1)
        attention = self.softmax(attention)
        attention_value = attention @ v  # (data,1,patch_dim)
        ycls = cls + attention_value
        ycls = self.gl(ycls)  # (data,patch+1,cls_dim)
        return ycls


model = CrossAttention(5, 6, 20)
cls = torch.randn([1, 1, 5])
patch = torch.randn([1, 1, 6])
