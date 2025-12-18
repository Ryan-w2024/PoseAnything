# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


class AELoss(nn.Module):
    """Associative Embedding loss. This loss is recommended for most grouping 
    including grouping graph id in Blumnet.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`
    """

    def __init__(self, loss_type='exp', push_loss_factor=1, pull_loss_factor=1):
        super().__init__()
        assert loss_type in ['exp', 'max']
        self.loss_type = loss_type
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor

    def singleTagLoss(self, pred_tag, gt_tag):
        """Associative embedding loss for one image.

        Args:
            pred_tag (torch.Tensor[N,]): tag channels of output.
            gt_tags (torch.Tensor[N,]): tag channels of gt.
        """
        gt_tag = gt_tag.int()
        max_bid = torch.max(gt_tag).cpu().data.numpy()
        tags = []
        pull = 0
        for per_bid in range(-1, max_bid + 1):
            same_tag = pred_tag[gt_tag == per_bid]
            if len(same_tag) == 0:
                continue
            tags.append(torch.mean(same_tag, dim=0))
            pull = pull + torch.mean((same_tag - tags[-1].expand_as(same_tag))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, gt_tags):
        """Accumulate the tag loss for each image in the batch.

        Note:
            batch_size: B

        Args:
            pred_tags (torch.Tensor[BxN]): tag channels of output.
            gt_tags (torch.Tensor[BxN]): tag channels of gt.
        """
        pushes, pulls = [], []
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], gt_tags[i])
            pushes.append(push)
            pulls.append(pull)

        return (torch.stack(pushes) * self.push_loss_factor,
                torch.stack(pulls) * self.pull_loss_factor)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class GraphIDLoss(nn.Module):
    """A simple graph id loss which also works like the Associative Embedding loss. 
    We use this loss only for fast trying. 
    """

    def __init__(self):
        super().__init__()

    def singleTagLoss(self, pred_tag, gt_tag):
        """Associative embedding loss for one image.

        Args:
            pred_tag (torch.Tensor[N,]): tag channels of output.
            gt_tags (torch.Tensor[N,]): tag channels of gt.
        """
        l2_regress = F.mse_loss(pred_tag, gt_tag)
        exp_regress = torch.mean(torch.exp((pred_tag - gt_tag).pow(2))) / pred_tag.shape[0]
        gid_loss = l2_regress + exp_regress
        return gid_loss

    def forward(self, tags, gt_tags):
        """Accumulate the tag loss for each image in the batch.

        Note:
            batch_size: B

        Args:
            pred_tags (torch.Tensor[BxN]): tag channels of output.
            gt_tags (torch.Tensor[BxN]): tag channels of gt.
        """
        gid_losses = []
        batch_size = tags.size(0)
        for i in range(batch_size):
            gid_loss = self.singleTagLoss(tags[i], gt_tags[i])
            gid_losses.append(gid_loss)

        return [torch.stack(gid_losses)]