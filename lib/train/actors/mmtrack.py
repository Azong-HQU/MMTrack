import copy
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

import numpy
from lib.apis.acc_eval import accuracy
from lib.train.admin import multigpu
import pycocotools.mask as maskUtils


class MMTrackActor(BaseActor):
    """ Actor for training VLTrack models """

    def __init__(self, net, objective, settings, loss_weight=None, cfg=None,
                det_coord=[0], det_coord_weight=1.5):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
            search_anno:   (x1, y1, w, h)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        with_bbox = self.cfg.TRAIN.BBOX_TASK
        with_mask = getattr(self.cfg.TRAIN, 'MASK_TASK', False)
        with_language = getattr(self.cfg.TRAIN, 'LANGUAGE_TASK', False)
        
        # forward pass
        out_dict = self.forward_pass(data, with_bbox=with_bbox, with_mask=with_mask, with_language=with_language)
        
        # compute losses
        loss, status = self.compute_losses(out_dict['logits'], out_dict['targets'], 
                                            aux_outputs=out_dict['aux_outputs'], 
                                            with_bbox=with_bbox, with_mask=with_mask)
        
        # compute ACC
        if with_bbox and with_mask:
            bbox_acc, mask_iou, mask_acc = self.calc_accuracy(out_dict, bbox_anno=data['search_anno'], mask_anno=data["search_masks"])
        elif with_bbox:
            bbox_acc, mask_iou, mask_acc = self.calc_accuracy(out_dict, bbox_anno=data['search_anno'])
        elif with_mask:
            bbox_acc, mask_iou, mask_acc = self.calc_accuracy(out_dict, mask_anno=data["search_masks"])

        if with_bbox:
            status['BboxAcc@0.5'] = bbox_acc
        
        if with_mask:
            status['mIoU'] = mask_iou  # IoU: 0 ~ 100
            status['MaskAcc@0.5'] = mask_acc[0]
            status['MaskAcc@0.6'] = mask_acc[1]
            status['MaskAcc@0.7'] = mask_acc[2]
            status['MaskAcc@0.8'] = mask_acc[3]
            status['MaskAcc@0.9'] = mask_acc[4]

        return loss, status

    def forward_pass(self, data, with_bbox=False, with_mask=False, with_language=False):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_masks = data['search_masks'][0].view(-1, *data['search_masks'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])  # (B, 64): center point = 1

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        if with_bbox and with_mask:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],  # attention mask
                                search_segmask_vertices=data['search_mask_vertices'][0], # segmentation mask vertices
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_mask:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                search_attn_mask=data['search_att'][0],
                                search_segmask_vertices=data['search_mask_vertices'][0],
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_bbox and with_language:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],
                                exp_str = data['exp_str'],               # language descriptions
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        elif with_bbox:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                search_anno=data['search_anno'],
                                search_attn_mask=data['search_att'][0],
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)

        return out_dict

    def compute_losses(self, logits, targets, aux_outputs=None, with_bbox=False, with_mask=False, return_status=True):
        """Args:
            logits (tensor): [batch_size, 1+4 or 1+2*num_ray, vocab_size].
            target (tensor): [batch_size, 1+4 or 1+2*num_ray].
        """
        batch_size, num_token = logits.size()[:2]

        if with_bbox and with_mask:
            weight = logits.new_ones((batch_size, num_token))
            overlay = [self.det_coord_weight if i % 5 in self.det_coord else 1. for i in range(6)] # 这里需要再检查一下是否正确,原始是5
            overlay = torch.tensor(overlay, device=weight.device, dtype=weight.dtype)
            for elem in weight:
                elem[:6] = overlay
            weight = weight.reshape(-1)
        elif with_bbox:
            weight = logits.new_tensor([self.det_coord_weight if i % 5 in self.det_coord else 1.
                                        for i in range(batch_size * num_token)])
        elif with_mask:
            num_bin = self.net.module.num_bin if multigpu.is_multi_gpu(self.net) else self.net.num_bin
            weight = logits.new_tensor([1. for _ in range(batch_size * num_token)])
            weight[targets.view(-1) == num_bin] /= 10.       # 这里需要再检查一下是否正确

        loss = []
        loss_status = {}
        if len(aux_outputs) > 0:
            for i in range(len(aux_outputs)):
                loss.append(self.objective['ce_loss'](aux_outputs[i]['logits'], targets, weight=weight))
            aux_loss = sum(loss) / len(loss)
            loss_status["Loss/aux"] = aux_loss.item()

        ce_loss = self.objective['ce_loss'](logits, targets, weight=weight)
        loss.append(ce_loss)

        if return_status:
            # status for log
            loss_status["Loss/cls"] = ce_loss.item()
            loss_status["Loss/total"] = sum(loss).item()

            return sum(loss), loss_status
        else:
            return sum(loss)

    def calc_accuracy(self, out_dict, bbox_anno=None, mask_anno=None):
        with_bbox = bbox_anno is not None
        with_mask = mask_anno is not None

        gt_bbox, gt_mask, is_crowd = None, [], []
        det_acc_list, mask_iou_list, mask_acc_list = [], [], []
        
        if with_bbox:
            gt_bbox = box_xywh_to_xyxy(bbox_anno)[0]  # norm coords: (x1,y1,w,h) to (X1, Y1, X2, Y2)
        
        if with_mask:
            # encode mask anno
            search_masks = numpy.ceil(mask_anno[0].cpu().numpy()) # .astype("uint8") # mask 这里转型有误差 (float to uint8)
            for mask in search_masks:
                mask_rle = maskUtils.encode(numpy.asfortranarray(mask, dtype=numpy.uint8))
                if isinstance(mask_rle['counts'], bytes):
                        mask_rle['counts'] = str(mask_rle['counts'], encoding='utf-8')
                
                # if len(mask_rle) > 1:
                #     is_crowd.append(1) # sometimes there are multiple binary map (corresponding to multiple segs)
                #     mask_rle = maskUtils.merge(mask_rle)
                # else:
                #     is_crowd.append(0)
                is_crowd.append(0)
                gt_mask.append(mask_rle)

        with torch.no_grad():
            if with_bbox and with_mask:
                batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                                pred_bboxes=out_dict['pred_bboxes'] * self.settings.output_sz['search'], # norm coord --> abs coord
                                gt_bbox=gt_bbox * self.settings.output_sz['search'],                 # norm coord --> abs coord
                                pred_masks=out_dict['pred_masks'],
                                gt_mask=gt_mask,
                                is_crowd=is_crowd, 
                                device=out_dict['pred_bboxes'].device)
            elif with_bbox:
                batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                                pred_bboxes=out_dict['pred_bboxes'] * self.settings.output_sz['search'],
                                gt_bbox=gt_bbox * self.settings.output_sz['search'],
                                device=out_dict['pred_bboxes'].device)
            elif with_mask:
                batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                                pred_masks=out_dict['pred_masks'],
                                gt_mask=gt_mask,
                                is_crowd=is_crowd, 
                                device=out_dict['targets'].device)

        if with_bbox:
            det_acc_list.append(batch_det_acc.item())
            bbox_acc = sum(det_acc_list) / len(det_acc_list)
        
        if with_mask:
            mask_iou_list.append(batch_mask_iou)
            mask_acc_list.append(batch_mask_acc_at_thrs)

            mask_iou = torch.cat(mask_iou_list).mean().item()
            mask_acc = torch.vstack(mask_acc_list).mean(dim=0).tolist()

        if with_bbox and with_mask:
            return bbox_acc, mask_iou, mask_acc
        elif with_bbox:
            return bbox_acc, None, None
        elif with_mask:
            return None, mask_iou, mask_acc