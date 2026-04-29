"""
object-freezer custom nodes.

Exports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS for ComfyUI.
"""

from .nodes.blueprint_creator import BlueprintCreator
from .nodes.blueprint_injector import BlueprintInjector
from .nodes.blueprint_outputs import BlueprintPathOutput

NODE_CLASS_MAPPINGS = {
    # Main node
    "BlueprintCreator": BlueprintCreator,
    "BlueprintInjector": BlueprintInjector,
    "BlueprintPathOutput": BlueprintPathOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueprintCreator": "Blueprint Creator (.blueprint)",
    "BlueprintInjector": "Blueprint Injector (patch model)",
    "BlueprintPathOutput": "Blueprint Path (Output)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']