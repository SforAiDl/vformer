"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""


class _ModelZooConfigs:

    MODEL_NAME_TO_CONFIG = {
        "Vanilla": "vit_tiny_patch16_224.py",
        "Swin": "swin_base_patch4_window7_224.py",
    }

    @staticmethod
    def query(model_name):

        if model_name in _ModelZooConfigs.MODEL_NAME_TO_CONFIG:

            cfg_file = _ModelZooConfigs.MODEL_NAME_TO_CONFIG[model_name]
            return cfg_file

        raise ValueError(f"Model name '{model_name}' not found in model zoo")
