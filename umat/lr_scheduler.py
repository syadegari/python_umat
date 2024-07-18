import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomCosineAnnealingWarmRestarts(_LRScheduler):
    """
    Similar ro ConsineAnnealingWarmRestart but the peak warm restart is reduced linearly from
    `initial_lr` to `final_lr`. After reaching `final_lr` it behaves like a normal ConsineAnnealingWarmRestart.
    We can also achieve constant lr after reaching `final_lr` by setting `eta_min` equal to `final_lr`.
    """

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        initial_lr=1e-3,
        final_lr=1e-5,
        restarts_until_final=10,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_T_0 = T_0
        self.current_epoch = last_epoch

        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.restarts_until_final = restarts_until_final
        self.lr_decay_step = (initial_lr - final_lr) / restarts_until_final

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == -1:
            return [self.initial_lr for _ in self.optimizer.param_groups]

        cycle = 0
        accumulated_epochs = 0
        while accumulated_epochs <= self.last_epoch:
            accumulated_epochs += self.base_T_0 * (self.T_mult**cycle)
            cycle += 1
        cycle -= 1

        self.T_0 = self.base_T_0 * (self.T_mult**cycle)

        if cycle > self.restarts_until_final:
            max_lr = self.final_lr
        else:
            max_lr = self.initial_lr - cycle * self.lr_decay_step

        cycle_start = accumulated_epochs - self.T_0
        cos_inner = (self.last_epoch - cycle_start) / self.T_0
        cos_out = self.eta_min + (max_lr - self.eta_min) * (1 + torch.cos(torch.tensor(torch.pi * cos_inner))) / 2

        return [cos_out.item() for _ in self.optimizer.param_groups]
