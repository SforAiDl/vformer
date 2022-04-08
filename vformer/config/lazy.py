from collections import abc
from dataclasses import is_dataclass
import os
import pkg_resources
from omegaconf import DictConfig, ListConfig

from .config_utils import _convert_target_to_string
# copied from detectron 2


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.
    Examples:
    ::
        from detectron2.config import instantiate, LazyCall
        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags={"allow_objects": True})
