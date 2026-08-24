"""Config conversion at the wiring boundary."""
from types import SimpleNamespace

from omegaconf import OmegaConf


def convert_cfg_to_native(cfg_obj):
    """
    Konvertiert OmegaConf rekursiv in native SimpleNamespace-Objekte.
    Dies ermöglicht C-Level Zugriffsgeschwindigkeiten auf Attribute.
    """
    if OmegaConf.is_config(cfg_obj):
        d = OmegaConf.to_container(cfg_obj, resolve=True)
    else:
        d = cfg_obj

    if isinstance(d, dict):
        return SimpleNamespace(**{k: convert_cfg_to_native(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_cfg_to_native(i) for i in d]
    
    return d
