class BlueprintPathOutput:
    """
    Minimal output node so workflows that only produce a blueprint file
    are considered to have an output in ComfyUI.
    """

    CATEGORY = "blueprint"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"blueprint_path": ("STRING", {"default": ""})}}

    # Match core node behavior (see comfy_extras/nodes_preview_any.py):
    # output node should still return a result value for UI display.
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("blueprint_path",)
    FUNCTION = "out"

    def out(self, blueprint_path):
        value = str(blueprint_path)
        return {"ui": {"text": (value,)}, "result": (value,)}

