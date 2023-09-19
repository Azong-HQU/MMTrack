import math
import random
import numpy as np
import os
from typing import List
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from lib.models.mmtrack.vit import vit_base_patch16_224
from lib.models.mmtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xywh_to_xyxy, box_xywh_to_cxcywh
from lib.models.transformers import build_decoder, VisionLanguageFusionModule, PositionEmbeddingSine1D
from lib.models.predictor import build_predictor
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.utils.misc import NestedTensor
from einops import repeat


class MMTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, encoder, decoder, predictor,
                tokenizer=None, text_encoder=None,
                feat_sz=20, num_bin=1000, shuffle_fraction=-1,
                top_p=-1, input_size=256,
                bbox_task=False, language_task=False,
                num_vlfusion_layers=0, vl_input_type='separate',
                bbox_type='xyxy', aux_loss=False):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.bbox_task = bbox_task
        self.top_p = top_p
        self.num_bin = num_bin
        self.shuffle_fraction = shuffle_fraction
        self.bbox_type = bbox_type
        self.backbone = encoder
        self.bottleneck = nn.Sequential(
                                nn.Linear(encoder.embed_dim, predictor.input_dim),
                                nn.GELU(),
                            )  # the bottleneck layer
        
        # text encoder
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.decoder = decoder

        # Auxiliary head
        self.aux_loss = aux_loss
        if aux_loss:
            self.predictor = _get_clones(predictor, decoder.num_decoder_layers)
        else:
            # self.predictor = _get_clones(predictor, 1)
            self.predictor = predictor
        
        text_feat_size = self.text_encoder.config.hidden_size
        
        self.text_adj = nn.Sequential(
            nn.Linear(text_feat_size, predictor.input_dim, bias=True),
            nn.LayerNorm(predictor.input_dim, eps=1e-12),
            nn.Dropout(0.1),
        )

        # multi-modal vision language fusion
        self.vl_fusion = VisionLanguageFusionModule(dim=predictor.input_dim, num_heads=8, attn_drop=0.1, proj_drop=0.1,
                                                    num_vlfusion_layers=num_vlfusion_layers, vl_input_type=vl_input_type)
        self.text_pos = PositionEmbeddingSine1D(predictor.input_dim, normalize=True)
        
        self.input_size = input_size
        self.feat_sz_s = int(feat_sz)
        self.feat_len_s = int(feat_sz ** 2)

        if bbox_task or language_task:
            # bbox_token, x1, y1, x2, y2, language_token, token1, token2
            self.task_embedding = nn.Embedding(1, predictor.input_dim)

    def quantize(self, seq):
        return (seq * self.num_bin).long()

    def dequantize(self, seq):
        return seq / self.num_bin

    def sequentialize(self,
                      gt_bbox=None,
                      text_embed=None,
                      bbox_type='xyxy',
                      ):
        """Args:
            gt_bbox (list[tensor]): [4, ]. (x1,y1,w,h)
            text_embed: (B, C)
        """
        with_bbox = gt_bbox is not None

        # text_embed repeat
        if with_bbox and not text_embed is None:
            text_embed = repeat(text_embed, 'b c -> b n c', n = 1) # (B, 256) to (B, n, 256)
            # text_embed = text_embed.unsqueeze(1)

        batch_size = gt_bbox.size(1)           # (1, b, 4)
        
        # quantize bbox coords
        if bbox_type == 'xyxy':
            seq_in_bbox = box_xywh_to_xyxy(gt_bbox)[0]    # (x1,y1,w,h) to (x1, y1, x2, y2): (B,4)
        elif bbox_type == 'cxcywh':
            seq_in_bbox = box_xywh_to_cxcywh(gt_bbox)[0]  # (x1,y1,w,h) to (cx, cy, w, h): (B,4)

        seq_in_bbox = self.quantize(seq_in_bbox)    # normalized coord -> quantize coord
        seq_in_bbox = seq_in_bbox.clamp(min=0, max=self.num_bin-1)

        seq_in = seq_in_bbox

        seq_in_embeds_bbox = self.decoder.query_embedding(seq_in)  # generate coordinates-based querys: (B,4,256)
        
        if not text_embed is None:
            task_language = self.task_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds_text = text_embed * task_language
            seq_in_embeds = torch.cat([seq_in_embeds_text, seq_in_embeds_bbox], dim=1)  # language + vision

        else:
            seq_in_embeds = torch.cat(
                [seq_in_embeds_bbox.new_zeros((batch_size, 1, self.decoder.d_model)), seq_in_embeds_bbox], dim=1)
        
        targets = torch.cat(
            [seq_in, seq_in.new_full((batch_size, 1), self.num_bin)], dim=-1)
        
        return seq_in_embeds, targets

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]

            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state
            text_features = self.text_adj(text_features)
            text_features = NestedTensor(text_features, text_attention_mask)

            text_sentence_features = encoded_text.pooler_output
            text_sentence_features = self.text_adj(text_sentence_features)
        else:
            raise ValueError("Please make sure the caption is a list of string")
        return text_features, text_sentence_features

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                search_attn_mask: torch.Tensor,
                search_anno=None,
                search_segmask_vertices=None,
                exp_str=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        """
        search_anno: (x1,y1,w,h)
        exp_str List[str]: language descriptions
        """
        with_bbox = search_anno is not None
        with_mask = search_segmask_vertices is not None

        # forward Language branch
        if exp_str:
            text_features, text_sentence_features = self.forward_text(exp_str, device=search.device)

        # forward Visual branch
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)  # x: (B, len_z + len_x, C)
        
        # encoder output for the search region (B, HW, C)
        enc_opt = x[:, -self.feat_len_s:]

        # adjust dims of visual features
        enc_opt = self.bottleneck(enc_opt)

        # generate position encoding for the encoder output and text features
        enc_mask, enc_pos_embeds = self.decoder.memory_mask_pos_enc(search_attn_mask, self.feat_sz_s)
        text_pos = self.text_pos(text_features)  # [batch_size, length, c]

        # Vision Language Multi-Modal Fusion
        vl_features = self.vl_fusion(enc_opt,
                                    text_features.tensors,
                                    query_pos=enc_pos_embeds,
                                    memory_pos=text_pos,
                                    query_key_padding_mask=enc_mask,
                                    memory_key_padding_mask=text_features.mask,
                                    need_weights=False)

        # generate coordinates-based querys and labels
        seq_in_embeds, targets = self.sequentialize(gt_bbox=search_anno,
                                                    text_embed=text_sentence_features, 
                                                    bbox_type=self.bbox_type)

        # Forward decoder
        logits, aux_logits = self.decoder(seq_in_embeds, vl_features, enc_mask, enc_pos_embeds, 
                                        return_intermediate_output=self.aux_loss)

        aux_outputs = []
        if self.aux_loss:
            for i in range(len(self.predictor)-1):
                if aux_logits[i].size(1) > 5:
                    aux_logits = aux_logits[:, :5]
                aux_logits[i] = self.predictor[i](aux_logits[i])

                aux_preds = self.train_statis(aux_logits[i], with_bbox=with_bbox, with_mask=with_mask)
                aux_outputs.append({
                                'logits': aux_logits[i],
                                'pred_bboxes': aux_preds['pred_bboxes'],
                                'pred_masks': aux_preds['pred_masks'],
                            })
        
        if logits.size(1) > 5:
            logits = logits[:, :5] # split language and vision: (B, 7, 256) to (B, 5, 256)
        # logits = self.predictor[-1](logits)  # Forward predictor
        logits = self.predictor(logits)  # Forward predictor
        # training statistics
        preds = self.train_statis(logits, with_bbox=with_bbox, with_mask=with_mask)

        out = {
                'logits': logits,
                'aux_outputs': aux_outputs,
                'targets': targets,
                'pred_bboxes': preds['pred_bboxes'],
                'pred_masks': preds['pred_masks'],
            }
        return out

    def train_statis(self, logits, with_bbox=False, with_mask=False, inference=False):
        # training statistics
        with torch.no_grad():
            if with_mask and with_bbox:
                logits_bbox = logits[:, :4, :-1]
                scores_bbox = F.softmax(logits_bbox, dim=-1)
                _, seq_out_bbox = scores_bbox.max(dim=-1, keepdim=False)

                logits_mask = logits[:, 5:-1, :-1]  # (B, bbox+mask, 1001) to (B, mask, 1000)
                scores_mask = F.softmax(logits_mask, dim=-1)
                _, seq_out_mask = scores_mask.max(dim=-1, keepdim=False)

                seq_out_dict = dict(seq_out_bbox=seq_out_bbox.detach(), seq_out_mask=seq_out_mask.detach())
            else:
                if with_bbox:
                    logits = logits[:, :-1, :-1]
                elif with_mask:
                    logits = logits[:, :-1, :-1]

                scores = F.softmax(logits, dim=-1)
                _, seq_out = scores.max(dim=-1, keepdim=False)

                if with_bbox:
                    seq_out_dict = dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    seq_out_dict = dict(seq_out_mask=seq_out.detach())

            return self.get_predictions(seq_out_dict, inference=inference)


    def get_predictions(self, seq_out_dict, inference=False, rescale=False):
        """Args:
            seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        """
        pred_bboxes, pred_masks = None, None
        with_bbox = 'seq_out_bbox' in seq_out_dict
        with_mask = 'seq_out_mask' in seq_out_dict

        if with_mask:
            seq_out_mask = seq_out_dict['seq_out_mask']
            seq_out_mask = seq_out_mask.cpu().numpy()
            pred_masks = []
            for pred_mask in seq_out_mask:
                if len(pred_mask) % 2 != 0:  # must be even
                    pred_mask = pred_mask[:-1]

                pred_mask = self.dequantize(pred_mask)
                pred_mask *= self.input_size  # norm coords --> input image coords
                # pred_mask = pred_mask.astype(np.float64)

                if len(pred_mask) < 4:  # at least three points to make a mask
                    pred_mask = np.array([0, 0, 0, 0, 0, 0], order='F', dtype=np.uint8)
                    pred_mask = [pred_mask]
                elif len(pred_mask) == 4:
                    pred_mask = pred_mask[None]  # as a bbox
                else:
                    pred_mask = [pred_mask]  # as a polygon

                pred_rles = maskUtils.frPyObjects(pred_mask, self.input_size, self.input_size)  # list[rle]
                pred_rle = maskUtils.merge(pred_rles)

                if inference:
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], self.input_size, self.input_size)
                    pred_mask = pred_mask.masks[0]
                    pred_masks.append(pred_mask)
                else:
                    pred_masks.append(pred_rle)

        if with_bbox:
            seq_out_bbox = seq_out_dict['seq_out_bbox']

            pred_bboxes = self.dequantize(seq_out_bbox)
            pred_bboxes = pred_bboxes.double()

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)


    def forward_encoder(self,
                        template: torch.Tensor,
                        search: torch.Tensor,
                        ce_template_mask=None,
                        ce_keep_rate=None,
                        return_last_attn=False,):
        x, _ = self.backbone(z=template, x=search,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=return_last_attn, )
        
        # encoder output for the search region (B, HW, C)
        enc_opt = x[:, -self.feat_len_s:]
        
        # adjust dims
        enc_opt = self.bottleneck(enc_opt)

        return enc_opt
    

    def forward_decoder(self, 
                        enc_opt: torch.Tensor,
                        search_attn_mask: torch.Tensor,
                        gt_mask_vertices=None,
                        text_features=None,
                        text_sentence_features=None,
                        with_bbox=False,
                        with_mask=False,
                        inference=False):
        
        h, w = search_attn_mask.shape[-2:] # search shape

        # generate position encoding for the encoder output
        enc_mask, enc_pos_embeds = self.decoder.memory_mask_pos_enc(search_attn_mask, self.feat_sz_s)
        text_pos = self.text_pos(text_features)  # [batch_size, length, c]

        # vision language early-fusion
        vl_features = self.vl_fusion(enc_opt,
                                    text_features.tensors,
                                    query_pos=enc_pos_embeds,
                                    memory_pos=text_pos,
                                    memory_key_padding_mask=text_features.mask)

        # decode coordinates-based querys
        seq_out_dict = self.generate_sequence(vl_features, enc_mask, enc_pos_embeds,
                                        text_embed=text_sentence_features,
                                        with_bbox=with_bbox, with_mask=with_mask)
        
        preds = self.get_predictions(seq_out_dict, inference=inference)

        return preds

    def generate_sequence(self, memory, memory_mask, memory_pos_embeds, text_embed=None, with_bbox=False, with_mask=False):
        """Args:
            memory (tensor): encoder's output, [batch_size, h*w, d_model].

            x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
                ignored position.

            x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        with_language = text_embed is not None

        batch_size = memory.size(0)
        
        if with_language:
            task_language = self.task_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            text_embed = text_embed.unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds = text_embed * task_language
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.decoder.d_model))  # (B, 1, 256)
        
        if with_mask:
            decode_steps = self.num_ray * 2 + 1
        elif with_bbox:
            decode_steps = 4
        
        seq_out = self.generate(seq_in_embeds, memory, memory_pos_embeds, memory_mask, decode_steps)
        
        if with_bbox:
            return dict(seq_out_bbox=seq_out)
        elif with_mask:
            return dict(seq_out_mask=seq_out)


    def generate(self, seq_in_embeds, memory, memory_pos_embeds, memory_mask, decode_steps):
        seq_out = []
        for step in range(decode_steps):
            # Forward decoder
            out = self.decoder(seq_in_embeds, memory, memory_mask, memory_pos_embeds, is_inference=True)  # (B, 1, 256)
            logits = out[:, -1, :]

            # Forward predictor
            logits = self.predictor(logits)  # (B, 1001)

            logits = logits[:, :-1]

            probs = F.softmax(logits, dim=-1)
            if self.top_p > 0.:
                sorted_score, sorted_idx = torch.sort(probs, descending=True)
                cum_score = sorted_score.cumsum(dim=-1)
                sorted_idx_to_remove = cum_score > self.top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove)
                probs = probs.masked_fill(idx_to_remove, 0.)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = probs.max(dim=-1, keepdim=True)

            seq_in_embeds = torch.cat(
                [seq_in_embeds, self.decoder.query_embedding(next_token)], dim=1)

            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, dim=-1)

        return seq_out

    def forward_text_encoder(self, exp_str, device):
        text_features, text_sentence_features = self.forward_text(exp_str, device=device)

        return text_features, text_sentence_features

    def track(self, template: torch.Tensor,
                search,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gt_mask_vertices=None,
                text_features=None,
                text_sentence_features=None,
                with_bbox=False,
                with_mask=False,
                inference=False,
                ):
        enc_opt = self.forward_encoder(template=template,
                                        search=search.tensors,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn)
        preds = self.forward_decoder(enc_opt=enc_opt,
                                    search_attn_mask=search.mask,
                                    gt_mask_vertices=gt_mask_vertices,
                                    text_features=text_features,
                                    text_sentence_features=text_sentence_features,
                                    with_bbox=with_bbox,
                                    with_mask=with_mask,
                                    inference=inference)
        return preds


def build_mmtrack(cfg, is_training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # Build Text Encoder
    tokenizer, text_encoder = None, None
    if cfg.MODEL.TEXT_ENCODER == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('pretrained_networks/roberta-base') # load pretrained RoBERTa Tokenizer
        text_encoder = RobertaModel.from_pretrained('roberta-base')  # load pretrained RoBERTa model
    elif cfg.MODEL.TEXT_ENCODER == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('pretrained_networks/bert-base-cased')
        text_encoder = BertModel.from_pretrained('bert-base-cased')
    
    decoder = build_decoder(cfg)
    predictor = build_predictor(cfg)

    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)

    language_task = getattr(cfg.TRAIN, "LANGUAGE_TASK", False)

    model = MMTrack(
        backbone,
        decoder,
        predictor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        feat_sz=feat_sz,
        input_size=cfg.DATA.SEARCH.SIZE,
        bbox_task=cfg.TRAIN.BBOX_TASK,
        language_task=language_task,
        num_vlfusion_layers=cfg.MODEL.VLFUSION_LAYERS,
        vl_input_type=cfg.MODEL.VL_INPUT_TYPE,
        bbox_type=cfg.MODEL.DECODER.BBOX_TYPE,
        aux_loss=cfg.TRAIN.AUX_LOSS,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and cfg.TRAIN.LANGUAGE_TASK and is_training:
        # Segmentation or Language Task: freeze viaual encoder, fine-tune decoder, query and head
        model_path = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print('Load OSTrack model from: ' + model_path)
        except:
            print("Warning: VLTrack model weights are not loaded !")

    return model
