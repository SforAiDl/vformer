<h1 align="center">VFormer</h1>
<h3 align="center">A modular PyTorch library for Vision Transformers</h3>

<div align='center'>

[![Build status](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml/badge.svg)](https://github.com/SforAiDl/vformer/actions/workflows/package-test.yml)

</div>





### References

- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

##   How to use?
####Vanilla Vision Transformer
```python
import torch
from vformer.models.classification import VanillaViT

model = VanillaViT(
        img_size=256,
        patch_size=32,
        n_classes=10,
        in_channels=3)
image=torch.randn(1,3,256,256)
predictions=model(image) #(1,10)
```
####Swin Transformer
```python
import torch
from vformer.models.classification import SwinTransformer

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
image=torch.randn(1,3,224,224)
predictions=model(image) #(1,10)
```



<br>
<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```bibtex
Vanilla Transformer
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
```bibtex
Swin Transformer
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

</details>
