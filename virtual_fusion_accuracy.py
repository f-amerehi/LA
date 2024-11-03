import torch
from torchmetrics import Metric


class VirtualFusionAccuracy(Metric):
    def __init__(self):
        # remember to call super
        super().__init__()
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lams: torch.Tensor) -> None:
        assert torch.Size([outputs.shape[0]]) == targets_a.shape
        assert targets_a.shape == targets_b.shape
        assert targets_a.shape == lams.shape
        _, preds = torch.max(outputs.data, 1)

        # update metric states
        correct = torch.mul(preds.eq(targets_a.data), lams).sum() + torch.mul(preds.eq(targets_b.data), (1 - lams)).sum()
        self.correct += correct.long()

        self.total += targets_a.numel()

    def compute(self) -> torch.Tensor:
        # compute final result
        return self.correct.float() / self.total
