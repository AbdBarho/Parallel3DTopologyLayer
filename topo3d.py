from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function
from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
import time
from torch.multiprocessing import get_context
from skimage.measure import label


def _get_slice_gradients(inputs: Tuple[Tensor, Tensor]):
    pred2d, gt2d = inputs
    if not gt2d.any():
        return torch.zeros_like(pred2d), torch.tensor(0.0)

    pred2d = pred2d.float()
    pred2d.requires_grad = True
    pred2d.retain_grad()

    _, k_gt = label(gt2d.numpy(), background=0,
                    connectivity=1, return_num=True)

    process_globals = globals()
    key = f'layer_{str(pred2d.shape)}'
    if key not in process_globals:
        process_globals[key] = LevelSetLayer2D(
            pred2d.shape, maxdim=0, sublevel=False)
    layer = process_globals[key]

    pred_barcode = layer(pred2d)
    top_k_layer_all_bars = TopKBarcodeLengths(dim=0, k=torch.numel(pred2d))
    non_zero_bars = torch.sum(top_k_layer_all_bars(pred_barcode) > 0.0).item()
    top_k_layer = TopKBarcodeLengths(dim=0, k=non_zero_bars)
    signs = torch.zeros(non_zero_bars)  # contains signs for bars
    # assign -1 to the number of bars that should be promoted by the loss
    signs[:k_gt] = 1.0

    pred_bars = top_k_layer(pred_barcode)
    losses = F.l1_loss(pred_bars, signs, reduction='none')
    loss = losses.sum()

    loss.backward()

    loss = loss.detach()
    grad = pred2d.grad.clone().detach()
    del pred2d, gt2d, pred_barcode, pred_bars
    return grad, loss


def get_grad_and_loss(patch3d: Tensor, mask3d: Tensor, pool):
    assert patch3d.shape == mask3d.shape
    # assert patch3d.shape[-2] == patch3d.shape[-1], "Must be square!!"
    it = pool.map(_get_slice_gradients, zip(
        patch3d.detach().cpu().contiguous(),
        mask3d.detach().cpu().to(torch.uint8).contiguous()
    ))
    grads = torch.stack([g for g, l in it], dim=0)
    losses = torch.stack([l for g, l in it], dim=0)
    return grads, losses


class _Topo3DFunc(Function):
    @staticmethod
    def forward(ctx, patch3d, mask3d, pool):
        grads, loss = get_grad_and_loss(patch3d, mask3d, pool)
        ctx.grads = grads  # .clip(-0.001, 0.001)
        ctx.device = patch3d.device
        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is always 1.0 in my testing
        return (grad_output * ctx.grads).to(ctx.device), None, None


class Topo3DLoss(torch.nn.Module):
    def __init__(self, num_processes: int) -> None:
        super().__init__()
        self.pool = get_context("spawn").Pool(num_processes)

    def forward(self, patch3d: Tensor, mask3d: Tensor) -> Tensor:
        return _Topo3DFunc.apply(patch3d, mask3d, self.pool)


class Topo3DLossRotatingAxis(Topo3DLoss):
    def __init__(self, num_processes: int) -> None:
        super().__init__(num_processes)
        self.axis = 0

    def forward(self, patch3d: Tensor, mask3d: Tensor) -> Tensor:
        axis = self.axis = (self.axis + 1) % 3
        if axis == 0:
            return super().forward(patch3d, mask3d)
        return super().forward(patch3d.transpose(0, axis), mask3d.transpose(0, axis))


class Topo3DLossAllAxes(Topo3DLossRotatingAxis):
    def forward(self, patch3d: Tensor, mask3d: Tensor) -> Tensor:
        f = super().forward
        l = f(patch3d, mask3d) + f(patch3d, mask3d) + f(patch3d, mask3d)
        return l / 3.0


def prepare_for_topology(y_pred: Tensor, y_true: Tensor):
    b, c, n, w, h = y_true.shape
    # assert c == 1 and y_true.shape == y_pred.shape

    r_pred = y_pred.reshape(-1, w, h)
    r_true = y_true.reshape(-1, w, h)

    # Force square crops, otherwise we would have problems!!
    # do random square crop
    assert h < w
    import random
    start_index = random.randint(0, w - h - 2)

    r_pred = r_pred[:, start_index: start_index + h]
    r_true = r_true[:, start_index: start_index + h]

    return r_pred, r_true
