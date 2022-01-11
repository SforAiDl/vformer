# thanks to https://github.com/isl-org/DPT

import math

import torch
import torch.nn.functional as F


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self.model._resize_pos_embed(
        self.model.pos_embedding.pos_embed,
        h // self.model.patch_size[1],
        w // self.model.patch_size[0],
    )

    B = x.shape[0]
    x = self.model.patch_embedding.patch_embedding(x)
    cls_tokens = self.model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.model.pos_embedding.pos_drop(x)
    x = self.model.encoder(x)
    return x


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb
