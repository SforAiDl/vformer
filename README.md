<h1 align="center">VFormer</h1>
<h3 align="center">A modular PyTorch library for Vision Transformers</h3>

<div align='center'>

[![Build status](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml/badge.svg)](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml)
[![codecov](https://codecov.io/gh/SforAiDl/vformer/branch/main/graph/badge.svg?token=5QKCZ67CM2)](https://codecov.io/gh/SforAiDl/vformer)


</div>

## Installation

```shell

git clone https://github.com/SforAiDl/vformer.git
cd vformer/
python setup.py install

```

## Example usage

To instantiate and use a Swin Transformer model -

```python

import torch
from vformer.models.classification import SwinTransformer

image = torch.randn(1, 3, 224, 224)       # Example data
model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_rate=0.2,
    )
logits = model(image)            
```
<br>

### References

- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [Attention-Visualization-Methods](https://github.com/jacobgil/vit-explain)

<br>

<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

<br>

<b>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</b>
```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```

<b>Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</b>
```bibtex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

</details>
