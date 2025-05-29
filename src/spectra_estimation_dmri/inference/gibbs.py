from ..registry import register


@register("inference", "gibbs")
class GibbsInference:
    def __init__(self, iterations, burn_in):
        self.iterations = iterations
        self.burn_in = burn_in

    def run(self, data, prior):
        # Placeholder for actual Gibbs sampling logic
        print(
            f"Running Gibbs for {self.iterations} iterations with burn-in {self.burn_in}"
        )
        return {"spectrum": [0.1, 0.2, 0.7]}  # Dummy result
