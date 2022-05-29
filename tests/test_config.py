import os
import tempfile
from itertools import count

import pytest
import torch
from omegaconf import DictConfig

from vformer.config import LazyCall
from vformer.config import LazyCall as L
from vformer.config import LazyConfig, instantiate
from vformer.models import PVTSegmentation, SwinTransformer, VanillaViT, ViViTModel2


def test_lazy():
    # classification models
    vanilla_config = LazyCall(VanillaViT)(img_size=224, patch_size=7, n_classes=10)
    swin_config = LazyCall(SwinTransformer)(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
    )
    vivit_config = LazyCall(ViViTModel2)(
        img_size=224,
        in_channels=3,
        patch_size=16,
        embedding_dim=192,
        depth=4,
        num_heads=3,
        head_dim=64,
        num_frames=1,
        n_classes=10,
    )

    # dense models
    pvt_config = LazyCall(PVTSegmentation)()
    pvt_config["img_size"] = 224
    rand_img_tensor = torch.randn(4, 3, 224, 224)
    rand_vdo_tensor = torch.randn([32, 16, 3, 224, 224])

    vanilla_vit = instantiate(vanilla_config)
    swin_vit = instantiate(swin_config)
    vivit = instantiate(vivit_config)

    pvt = instantiate(pvt_config)

    assert vanilla_vit(rand_img_tensor).shape == (4, 10)
    assert swin_vit(rand_img_tensor).shape == (4, 10)
    assert pvt(rand_img_tensor).shape == (4, 1, 224, 224)
    assert vivit(rand_vdo_tensor).shape == (32, 10)


def test_raise_errors():
    a = "strings"
    with pytest.raises(TypeError):
        LazyConfig(a)
    with pytest.raises(TypeError):
        LazyCall(2)

    cfg = [1, 2, 3, 4]
    cfg2 = instantiate(cfg)
    assert cfg2 == cfg, "it should return same object"

    with pytest.raises(AssertionError):
        instantiate({"_target_": "test"})


def test_load():
    root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")
    cfg = LazyConfig.load(root_filename)

    assert cfg.dir1a_dict.a == "modified"

    assert cfg.dir1b_dict.a == 1
    assert cfg.lazyobj.x == "base_a_1"

    cfg.lazyobj.x = "new_x"
    # reload
    cfg = LazyConfig.load(root_filename)
    assert cfg.lazyobj.x == "base_a_1"


def test_save_load():
    root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")
    cfg = LazyConfig.load(root_filename)
    with tempfile.TemporaryDirectory(prefix="vformer") as d:
        fname = os.path.join(d, "test_config.yaml")
        LazyConfig.save(cfg, fname)
        cfg2 = LazyConfig.load(fname)

    assert cfg2.lazyobj._target_ == "itertools.count"
    assert cfg.lazyobj._target_ == count
    cfg2.lazyobj.pop("_target_")
    cfg.lazyobj.pop("_target_")
    # the rest are equal
    assert cfg == cfg2


def test_failed_save():
    cfg = DictConfig({"x": lambda: 3}, flags={"allow_objects": True})
    with tempfile.TemporaryDirectory(prefix="vformer") as d:
        fname = os.path.join(d, "test_config.yaml")
        LazyConfig.save(cfg, fname)
        assert os.path.exists(fname) == True
        assert os.path.exists(fname + ".pkl") == True


def test_overrides():
    root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")

    cfg = LazyConfig.load(root_filename)
    LazyConfig.apply_overrides(cfg, ["lazyobj.x=123", 'dir1b_dict.a="123"'])
    assert cfg.dir1b_dict.a == "123"
    assert cfg.lazyobj.x == 123


def test_invalid_overrides():
    root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")

    cfg = LazyConfig.load(root_filename)
    with pytest.raises(KeyError):
        LazyConfig.apply_overrides(cfg, ["lazyobj.x.xxx=123"])


def test_to_py():
    root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")

    cfg = LazyConfig.load(root_filename)
    cfg.lazyobj.x = {
        "a": 1,
        "b": 2,
        "c": L(count)(x={"r": "a", "s": 2.4, "t": [1, 2, 3, "z"]}),
    }
    cfg.list = ["a", 1, "b", 3.2]
    py_str = LazyConfig.to_py(cfg)
    expected = """cfg.dir1a_dict.a = "modified"
cfg.dir1a_dict.b = 2
cfg.dir1b_dict.a = 1
cfg.dir1b_dict.b = 2
cfg.lazyobj = itertools.count(
    x={
        "a": 1,
        "b": 2,
        "c": itertools.count(x={"r": "a", "s": 2.4, "t": [1, 2, 3, "z"]}),
    },
    y="base_a_1_from_b",
)
cfg.list = ["a", 1, "b", 3.2]
"""
    assert py_str == expected

    root_filename = os.path.join(os.path.dirname(__file__), "testing.yaml")
    cfg = LazyConfig.load(root_filename)
    obj = LazyConfig.to_py(cfg)


def test_check_configs():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "Vanilla",
        "vit_tiny_patch16_224.py",
    )

    cfg = LazyConfig.load(config_dir)
    cfg.model.img_size = 224
    cfg.model.in_channels = 3
    cfg.model.n_classes = 1000

    new_model = instantiate(cfg.model)
    assert new_model(torch.randn(4, 3, 224, 224)).shape == (4, 1000)

    cfg.model.num_classes = 10
    with pytest.raises(TypeError):
        # this will throw an error because `num_class` is not an acceptable input keyword, we use `n_classes`
        new_model = instantiate((cfg.model))

    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        "Swin",
        "swin_base_patch4_window7_224.py",
    )
    cfg = LazyConfig.load(config_dir)
    new_model = instantiate(cfg.model)
    assert new_model(torch.randn(4, 3, 224, 224)).shape == (4, 1000)
