from vformer.attention import VanillaSelfAttention
from vformer.functional import PreNorm
from vformer.config import LazyCall as L

encoder = L(PreNorm(dim=96,
                    fn=L(VanillaSelfAttention)(dim=96, num_heads=3, head_dim=96, p_dropout = 0.2)))
