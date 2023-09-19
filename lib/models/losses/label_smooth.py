import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self,
                 neg_factor=0.1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.neg_factor = neg_factor
        self.reduction = 'mean'
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets, weight):
        logits = logits.float()
        batch_size, num_pts, num_classes = logits.size(0), logits.size(1), logits.size(2)
        logits = logits.reshape(-1, num_classes)  # (B,5,1001) to (B*5, 1001)
        targets = targets.reshape(-1, 1)  # (B*num_token,) to (B*num_token,1)

        with torch.no_grad():
            targets = targets.clone().detach()
            label_pos, label_neg = 1. - self.neg_factor, self.neg_factor / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(label_neg)
            
            lb_one_hot.scatter_(1, targets, label_pos)

            lb_one_hot = lb_one_hot.detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=batch_size*num_pts)

        return loss
