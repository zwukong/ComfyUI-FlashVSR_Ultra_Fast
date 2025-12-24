from .nodes import FlashVSRNode, FlashVSRNodeAdv, FlashVSRNodeInitPipe

# Node-ların sistemdəki (kod) adları - hamısının sonuna _Stable artırdım
NODE_CLASS_MAPPINGS = {
    "FlashVSR_Stable": FlashVSRNode,
    "FlashVSR_Stable_Adv": FlashVSRNodeAdv,
    "FlashVSR_Stable_Pipe": FlashVSRNodeInitPipe
}

# Node-ların menyuda görünən adları
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSR_Stable": "FlashVSR Ultra-Fast (Stable)",
    "FlashVSR_Stable_Adv": "FlashVSR Advanced (Stable)",
    "FlashVSR_Stable_Pipe": "FlashVSR Init Pipe (Stable)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']