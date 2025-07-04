import hashlib
import json
from omegaconf import OmegaConf


def set_spectra_id(signal_decay, exp_config):
    """
    Generate a unique spectra_id based on only the fields that influence the spectrum.
    Args:
        signal_decay: SignalDecay object
        exp_config: Hydra experiment config (OmegaConf)
    Returns:
        spectra_id: str (12-char md5 hash)
    """

    def to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    # Convert relevant parts of the config to plain dicts
    model_cfg = OmegaConf.to_container(exp_config.model, resolve=True)
    inference_cfg = OmegaConf.to_container(exp_config.inference, resolve=True)
    diff_values = OmegaConf.to_container(exp_config.dataset.diff_values, resolve=True)

    hash_dict = {
        "signal_values": to_serializable(signal_decay.signal_values),
        "b_values": to_serializable(signal_decay.b_values),
        "snr": signal_decay.snr,
        "model": model_cfg,
        "inference": inference_cfg,
        "diff_values": diff_values,
    }
    print(hash_dict)
    json_repr = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_repr.encode("utf-8")).hexdigest()[:12]
