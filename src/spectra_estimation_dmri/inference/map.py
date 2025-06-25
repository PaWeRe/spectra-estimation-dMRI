import numpy as np
import arviz as az


class MAPInference:
    def __init__(self, model, signal, config=None):
        self.model = model
        self.signal = signal
        self.config = config
        self.regularization = (
            getattr(config, "l2_lambda", 0.0) if config is not None else 0.0
        )

    def run(self, return_idata=True):
        # Compute MAP/NNLS estimate
        fractions = self.model.map_estimate(
            self.signal, regularization=self.regularization
        )
        result = {"map_estimate": fractions}
        if return_idata:
            # Wrap as ArviZ InferenceData for diagnostics/plotting
            idata = az.from_dict(
                posterior={
                    "R": fractions[None, :]  # shape (chain, draw, dim) or (draw, dim)
                }
            )
            result["idata"] = idata
        return result
