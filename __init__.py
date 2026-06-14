"""
ComfyUI-ZPhoton
Photorealism-focused sampling nodes for the Z-Image / Z-Image-Turbo model.
"""
from .zphoton.nodes import (NODE_CLASS_MAPPINGS as _NODES,
                            NODE_DISPLAY_NAME_MAPPINGS as _NAMES)
from .zphoton.lora_stack import (NODE_CLASS_MAPPINGS as _LORA_NODES,
                                 NODE_DISPLAY_NAME_MAPPINGS as _LORA_NAMES)
from .zphoton.autoqc import (NODE_CLASS_MAPPINGS as _QC_NODES,
                             NODE_DISPLAY_NAME_MAPPINGS as _QC_NAMES)

NODE_CLASS_MAPPINGS = {**_NODES, **_LORA_NODES, **_QC_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_NAMES, **_LORA_NAMES, **_QC_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
