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
    likelihood_cfg = OmegaConf.to_container(exp_config.likelihood, resolve=True)
    prior_cfg = OmegaConf.to_container(exp_config.prior, resolve=True)
    inference_cfg = OmegaConf.to_container(exp_config.inference, resolve=True)
    # spectrum_pair only exists for simulated data, not BWH
    spectrum_pair = getattr(exp_config.dataset, "spectrum_pair", None)

    hash_dict = {
        "signal_values": to_serializable(signal_decay.signal_values),
        "b_values": to_serializable(signal_decay.b_values),
        "snr": signal_decay.snr,
        "likelihood": likelihood_cfg,
        "prior": prior_cfg,
        "inference": inference_cfg,
        "spectrum_pair": spectrum_pair,
    }
    # Debug: print(hash_dict)
    json_repr = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_repr.encode("utf-8")).hexdigest()[:12]


def get_group_id(signal_decay, exp_config):
    """
    Generate a group id hash for grouping spectra by config and signal decay (excluding noise realization).
    Args:
        signal_decay: SignalDecay object
        exp_config: Hydra experiment config (OmegaConf)
    Returns:
        group_id: str (12-char md5 hash)
    """

    def to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    likelihood_cfg = OmegaConf.to_container(exp_config.likelihood, resolve=True)
    prior_cfg = OmegaConf.to_container(exp_config.prior, resolve=True)
    inference_cfg = OmegaConf.to_container(exp_config.inference, resolve=True)
    # spectrum_pair only exists for simulated data, not BWH
    spectrum_pair = getattr(exp_config.dataset, "spectrum_pair", None)

    hash_dict = {
        # Exclude signal_values!
        "b_values": to_serializable(signal_decay.b_values),
        "snr": signal_decay.snr,
        "likelihood": likelihood_cfg,
        "prior": prior_cfg,
        "inference": inference_cfg,
        "spectrum_pair": spectrum_pair,
    }
    json_repr = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_repr.encode("utf-8")).hexdigest()[:12]
