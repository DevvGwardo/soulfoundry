"""ComfyUI custom node entry point.

ComfyUI imports this file and reads NODE_CLASS_MAPPINGS. Keep it boring.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"
