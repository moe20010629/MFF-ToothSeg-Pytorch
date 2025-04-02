import torch
import torch.nn as nn
import torch.nn.functional as F


# pred = torch.randn((1, 16000, 33))
# target = torch.ones((1, 16000)).long()
# iou_tabel = torch.zeros((33,3))
# def compute_iou(pred,target,iou_tabel):  # pred [B,N,C] target [B,N]
#     # iou_list = []
#     # target = target.data.numpy()
#     for j in range(pred.size(0)):
#         batch_pred = pred[j]  # batch_pred [N,C]
#         batch_target = target[j]  # batch_target [N]
#         batch_choice = batch_pred.max(1)[1]  # index of max value  batch_choice [N]
#         for cat in torch.unique(batch_target):
#             # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
#             # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
#             # iou = intersection/union if not union ==0 else 1
#             I = torch.sum(torch.logical_and(batch_choice == cat, batch_target == cat))
#             U = torch.sum(torch.logical_or(batch_choice == cat, batch_target == cat))
#             if U == 0:
#                 iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             iou_tabel[cat,0] += iou
#             iou_tabel[cat,1] += 1
#             # iou_list.append(iou)
#     return iou_tabel

# iou_tabel = compute_iou(pred, target, iou_tabel)


# PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice



class GeoLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, gamma=2, r=0.4):
        super(GeoLoss, self).__init__()
        self.gamma = gamma
        self.r = r
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, curvatures, smooth=1):
        # Ensure inputs are in the range [0, 1]
        inputs = torch.sigmoid(inputs)

        # Flatten label, prediction, and curvature tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        curvatures = curvatures.view(-1)

        # Calculate the number of points
        num_points = inputs.size(0)

        # Get the indices of the top r*100% curvature points
        _, indices = torch.topk(curvatures, int(self.r * num_points), largest=True, sorted=False)
        top_curvature_mask = torch.zeros(num_points, dtype=torch.bool, device=inputs.device)
        top_curvature_mask[indices] = True

        # Get the predicted probabilities and true labels for the top curvature points
        top_inputs = inputs[top_curvature_mask]
        top_targets = targets[top_curvature_mask]

        # Calculate the Dice loss for the top curvature points
        intersection = (top_inputs * top_targets).sum()
        dice = (2. * intersection + smooth) / (top_inputs.sum() + top_targets.sum() + smooth)

        loss = 1 - dice

        # Apply weighting if provided
        if self.weight is not None:
            loss *= self.weight

        return loss
