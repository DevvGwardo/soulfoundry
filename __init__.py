"""SoulFoundry - open-weights character consistency for ComfyUI.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS. Node
implementations live in ``nodes/``; all real logic lives in ``nodes/core/``
so it can be tested without ComfyUI, torch, or a GPU.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.2.0"

# There is deliberately no WEB_DIRECTORY here. The pack ships no frontend
# extension, and pointing ComfyUI at a directory that does not exist makes the
# UI request a missing asset on every load. Add the constant back only when
# ``web/`` actually contains an extension file.
