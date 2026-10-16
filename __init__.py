"""SoulFoundry - open-weights character consistency for ComfyUI.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS. Node
implementations live in ``nodes/``; all real logic lives in ``nodes/core/``
so it can be tested without ComfyUI, torch, or a GPU.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.0"

WEB_DIRECTORY = "./web"
