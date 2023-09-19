import time
import torch
import numpy
import pycocotools.mask as maskUtils

from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
        gt_mask (list[RLE]):
        pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, is_crowd)
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou.copy())

    return mask_iou


def accuracy(pred_bboxes=None, gt_bbox=None, 
             pred_masks=None, gt_mask=None, is_crowd=None, device="cuda"):
    eval_det = pred_bboxes is not None
    eval_mask = pred_masks is not None

    det_acc = torch.tensor([-1.], device=device)
    bbox_iou = torch.tensor([-1.], device=device)
    if eval_det:
        bbox_iou = bbox_overlaps(gt_bbox, pred_bboxes, is_aligned=True)
        det_acc = (bbox_iou >= 0.5).float().mean()

    mask_iou = torch.tensor([-1.], device=device)
    mask_acc_at_thrs = torch.full((5, ), -1., device=device)
    if eval_mask:
        mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100., mask_iou * 100., mask_acc_at_thrs * 100.
