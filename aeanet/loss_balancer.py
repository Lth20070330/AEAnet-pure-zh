class ParetoLossAdjuster(object):
    """Thesis-style loss weighting based on distance from historical best loss."""

    def __init__(self, num_losses=4, min_weight=0.1, eps=1e-6):
        if min_weight * num_losses > 1.0:
            raise ValueError("min_weight * num_losses must not exceed 1")
        self.num_losses = num_losses
        self.min_weight = min_weight
        self.eps = eps
        self.best_losses = [float("inf")] * num_losses
        self.weights = [1.0 / num_losses] * num_losses

    def adjust_weights(self, current_losses):
        values = [float(x) for x in current_losses]
        if len(values) != self.num_losses:
            raise ValueError("Unexpected number of losses")
        for i, value in enumerate(values):
            if value < self.best_losses[i]:
                self.best_losses[i] = value

        deltas = [max(value - best, self.eps) for value, best in zip(values, self.best_losses)]
        total = sum(deltas)
        raw = [value / total for value in deltas]

        if any(weight < self.min_weight for weight in raw):
            needed = sum(max(self.min_weight - weight, 0.0) for weight in raw)
            excess = [max(weight - self.min_weight, 0.0) for weight in raw]
            excess_total = sum(excess)
            if excess_total <= self.eps:
                self.weights = [1.0 / self.num_losses] * self.num_losses
            else:
                self.weights = [
                    weight - (extra / excess_total) * needed
                    if weight >= self.min_weight
                    else self.min_weight
                    for weight, extra in zip(raw, excess)
                ]
        else:
            self.weights = raw
        normalization = sum(self.weights)
        self.weights = [weight / normalization for weight in self.weights]
        return list(self.weights)

    def state_dict(self):
        return {
            "num_losses": self.num_losses,
            "min_weight": self.min_weight,
            "eps": self.eps,
            "best_losses": list(self.best_losses),
            "weights": list(self.weights),
        }

    def load_state_dict(self, state):
        self.best_losses = list(state["best_losses"])
        self.weights = list(state["weights"])

