import torch
import torch.nn as nn
from einops import rearrange


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
    def __init__(self, cls_dim, patch_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.fl = Projection(cls_dim, patch_dim)
        self.gl = Projection(patch_dim, cls_dim)
        self.to_k = nn.Linear(patch_dim, inner_dim)
        self.to_v = nn.Linear(patch_dim, inner_dim)
        self.to_q = nn.Linear(patch_dim, inner_dim)
        self.cls_project = Projection(inner_dim, patch_dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, cls, patches):
        # cls (data,1,cls_dim) patch (data,patch,patch_dim)
        cls = self.fl(cls)
        x = torch.cat([cls, patches], dim=-2)
        q = self.to_q(cls)  # (data,1,inner_embedding)
        k = self.to_k(x)  # (data,patch+1,inner_embedding)
        v = self.to_v(x)  # (data,patch+1,inner_embedding)
        k = rearrange(
            k, "b n (h d) -> b h n d", h=self.heads
        )  # (data,head,patch+1,head_dim)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)  # (data,head,1,head_dim)
        v = rearrange(
            v, "b n (h d) -> b h n d", h=self.heads
        )  # (data,head,patch+1,head_dim)
        k = torch.transpose(k, -2, -1)  # (data,head,head_dim,patch+1)
        attention = (q @ k) * self.scale  # (data,head,1,patch+1)
        attention = self.attend(attention)
        attention_value = attention @ v  # (data,head,1,head_dim)
        attention_value = rearrange(
            attention_value, "b h n d -> b n (h d)"
        )  # (data,1,inner_embedding)
        attention_value = self.cls_project(attention_value)
        ycls = cls + attention_value
        ycls = self.gl(ycls)  # (data,1,cls_dim)
        return ycls
