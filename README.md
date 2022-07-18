<h1 align="center">VFormer</h1>
<h3 align="center">A modular PyTorch library for vision transformers models</h3>

<div align='center'>

[![Tests](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml/badge.svg)](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml)
[![Docs](https://readthedocs.org/projects/vformer/badge/?version=latest)](https://vformer.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SforAiDl/vformer/branch/main/graph/badge.svg?token=5QKCZ67CM2)](https://codecov.io/gh/SforAiDl/vformer)
[![Downloads](https://pepy.tech/badge/vformer)](https://pepy.tech/project/vformer)

**[Documentation](https://vformer.readthedocs.io/en/latest/)**

</div>

## Library Features

- Contains implementations of prominent ViT architectures broken down into modular components like encoder, attention mechanism, and decoder
- Makes it easy to develop custom models by composing components of different architectures
- Contains utilities for visualizing attention maps of models using techniques such as gradient rollout

## Installation

### From source (recommended)

```shell

git clone https://github.com/SforAiDl/vformer.git
cd vformer/
python setup.py install

```

### From PyPI

```shell

pip install vformer

```

## Models supported

- [x] [Vanilla ViT](https://arxiv.org/abs/2010.11929)
- [x] [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [x] [Pyramid Vision Transformer](https://arxiv.org/abs/2102.12122)
- [x] [CrossViT](https://arxiv.org/abs/2103.14899)
- [x] [Compact Vision Transformer](https://arxiv.org/abs/2104.05704)
- [x] [Compact Convolutional Transformer](https://arxiv.org/abs/2104.05704)
- [x] [Visformer](https://arxiv.org/abs/2104.12533)
- [x] [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)
- [x] [CvT](https://arxiv.org/abs/2103.15808)
- [x] [ConViT](https://arxiv.org/abs/2103.10697)
- [x] [ViViT](https://arxiv.org/abs/2103.15691)
- [x] [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [x] [Memory Efficient Attention](https://arxiv.org/abs/2112.05682)

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

`VFormer` has a modular design and allows for easy experimentation using blocks/modules of different architectures. For example, if desired, you can use just the encoder or the windowed attention layer of the Swin Transformer model.

```python

from vformer.attention import WindowAttention

window_attn = WindowAttention(
        dim=128,
        window_size=7,
        num_heads=2,
        **kwargs,
    )

```

```python

from vformer.encoder import SwinEncoder

swin_encoder = SwinEncoder(
        dim=128,
        input_resolution=(224, 224),
        depth=2,
        num_heads=2,
        window_size=7,
        **kwargs,
    )

```

Please refer to our [documentation](https://vformer.readthedocs.io/en/latest/) to know more.

<br>

### References

- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [PVT](https://github.com/whai362/PVT)
- [vit-explain](https://github.com/jacobgil/vit-explain)
- [CrossViT](https://github.com/IBM/CrossViT)
- [Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers)
- [Visformer](https://github.com/danczs/Visformer)
- [DPT](https://github.com/isl-org/DPT)
- [CvT](https://github.com/microsoft/CvT)
- [convit](https://github.com/facebookresearch/convit)
- [ViViT-pytorch](https://github.com/rishikksh20/ViViT-pytorch)
- [perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)
- [memory-efficient-attention](https://github.com/AminRezaei0x443/memory-efficient-attention)
<!-- <br>

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

<b>Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions</b>
```bibtex
@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions},
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<b> CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification </b>

```bibtex
@inproceedings{chen2021crossvit,
    title={{CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification}},
    author={Chun-Fu (Richard) Chen and Quanfu Fan and Rameswar Panda},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2021}
}
```

<b> Escaping the Big Data Paradigm with Compact Transformers </b>

```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

<b>Visformer: The Vision-friendly Transformer</b>

```bibtex
@misc{chen2021visformer,
      title={Visformer: The Vision-friendly Transformer},
      author={Zhengsu Chen and Lingxi Xie and Jianwei Niu and Xuefeng Liu and Longhui Wei and Qi Tian},
      year={2021},
      eprint={2104.12533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<b>Vision Transformers for Dense Prediction</b>

```bibtex
@misc{ranftl2021vision,
      title={Vision Transformers for Dense Prediction},
      author={René Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
      year={2021},
      eprint={2103.13413},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
</details> -->
