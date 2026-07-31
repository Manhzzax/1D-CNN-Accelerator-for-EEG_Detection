"""Group distributionally robust objective for source patient groups."""

import torch


class GroupDROObjective:
    """Exponentiated-gradient GroupDRO over observed source patient groups."""

    def __init__(self, group_count, eta):
        if group_count < 2:
            raise ValueError("GroupDRO requires at least two source patient groups")
        if not 0.0 < eta <= 1.0:
            raise ValueError("GroupDRO eta must be in (0, 1]")
        self.group_count = int(group_count)
        self.eta = float(eta)
        self._log_weights = None

    def _weights_on(self, device):
        if self._log_weights is None or self._log_weights.device != device:
            self._log_weights = torch.zeros(self.group_count, device=device, dtype=torch.float32)
        return torch.softmax(self._log_weights, dim=0)

    def __call__(self, per_sample_losses, group_labels):
        if per_sample_losses.ndim != 1 or group_labels.ndim != 1:
            raise ValueError("GroupDRO losses and group labels must be one-dimensional")
        if len(per_sample_losses) != len(group_labels):
            raise ValueError("GroupDRO losses and group labels must have equal length")
        if torch.any(group_labels < 0) or torch.any(group_labels >= self.group_count):
            raise ValueError("GroupDRO labels are outside the configured source patient groups")

        group_losses = []
        active_groups = []
        for group_index in range(self.group_count):
            mask = group_labels == group_index
            if torch.any(mask):
                active_groups.append(group_index)
                group_losses.append(per_sample_losses[mask].mean())
        if not active_groups:
            raise ValueError("GroupDRO requires at least one observed patient group per batch")

        active_groups = torch.tensor(active_groups, device=per_sample_losses.device, dtype=torch.long)
        group_losses = torch.stack(group_losses)
        weights = self._weights_on(per_sample_losses.device)
        with torch.no_grad():
            self._log_weights[active_groups] += self.eta * group_losses.detach().float()
            self._log_weights -= torch.logsumexp(self._log_weights, dim=0)
            weights = torch.softmax(self._log_weights, dim=0)
        active_weights = weights[active_groups]
        active_weights = active_weights / active_weights.sum()
        return torch.sum(active_weights * group_losses)

    def weights(self):
        if self._log_weights is None:
            return torch.full((self.group_count,), 1.0 / self.group_count)
        return torch.softmax(self._log_weights.detach(), dim=0)
