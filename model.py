import time

import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy
from thop import profile
import numpy as np
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import wandb
import pprint
import torchvision

import vitdet
from tqdm import tqdm
import sys
from utils import AverageMeter
import cv2
import os
import shutil
from losses import DiceLoss
from networks.curfnet import Curfres
from networks.LGC_Encoder import LGC_Encoder
from utils import IoU


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):#10,256(c),32,32|10,106,2->

        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape


        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0#10,106,2
        

        sample_points = sample_points.unsqueeze(2)#10,106,1,2
        

        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)#10,256(c),32,32|10,106,1,2->10,256,106,1
        

        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)#10,106,256
        return sampled_features
    

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )


        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid):


        point_features = F.relu(self.feature_proj(point_features))#(10, 106, 256)->(10, 106, 128)

        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2).type(torch.int64)# (10, 128, 16, 2)-> (10, 2048, 2)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)#(10, 2048)

        src_features = point_features[batch_indices, pairs[:, :, 0]]#(10, 2048,128)
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]#(10, 2048,128)
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0]]#(10, 2048,2)
        tgt_points = points[batch_indices, pairs[:, :, 1]]#(10, 2048,2)
        offset = tgt_points - src_points#(10, 2048,2)

        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, offset], dim=2)#(10, 2048,258)
        
        

        pair_features = F.relu(self.pair_proj(pair_features))#(10, 2048,128)

        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)#(1280, 16,128)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)#(1280, 16)
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)#(1280, 1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)#(1280, 16)

        padding_mask = ~pairs_valid#(1280, 16)

        if self.config.TOPONET_VERSION != 'no_transformer':
            temp_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)#(1280, 16,128)->(1280, 16,128)
            if temp_features.shape[1]!=pair_features.shape[1]:
                temp_n=pair_features.shape[1]-temp_features.shape[1]
                sequence_length=torch.zeros((pair_features.shape[0],temp_n,pair_features.shape[2])).to(pair_features.device)
                pair_features=torch.cat((temp_features,sequence_length),dim=1)
            else:
                pair_features=temp_features
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)#(10,128, 16,128)

        logits = self.output_proj(pair_features)#(10,128, 16,1)

        scores = torch.sigmoid(logits)

        return logits, scores



class _LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv



class MTRoad(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
        elif config.SAM_VERSION == 'vit_l':
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.SAM_VERSION == 'vit_h':
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###

        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16

        if self.config.DATA_FUSION_TYPE == "ITO":
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53,114.49,114.49]).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375,57.63,57.63]).view(-1, 1, 1), False)

        else:
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        #select network type
        if self.config.DATA_FUSION_TYPE == "ITO":  # channel=5
            in_chans = 5
        else:
            in_chans = 3

        self.lgc_encoder=LGC_Encoder(
                config,
                in_chans=in_chans,
                encoder_1dconv=0,
                decoder_1dconv=4,
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            )

        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
            self.mask_dice=DiceLoss()

        self.surface_criterion=torch.nn.BCEWithLogitsLoss()
        self.surface_dice=DiceLoss()

        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        self.metrics = IoU()


        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            if self.config.DATA_FUSION_TYPE=="ITO":
                nn.init.kaiming_normal_(self.lgc_encoder.cen_encoder.patch_embed.proj.weight,mode='fan_out')
                ckpt_state_dict['image_encoder.patch_embed.proj.weight']=self.lgc_encoder.cen_encoder.patch_embed.proj.weight
            else:
                pass

            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            new_ckpt_state_dict = {}
            if not config.ORIGINAL_DLINK_SAM_ROAD:
                for key,value in ckpt_state_dict.items():
                    if key.split('.')[0]=="image_encoder":
                        new_key=key.replace("image_encoder","lgc_encoder.cen_encoder")
                        new_ckpt_state_dict[new_key]=ckpt_state_dict[key]
                    else:
                        new_ckpt_state_dict[key]=ckpt_state_dict[key]

                ckpt_state_dict=new_ckpt_state_dict

            matched_names = []#177
            mismatch_names = []
            state_dict_to_load = {}
            a=self.named_parameters()
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict


    def forward(self, rgb):
        x = rgb.permute(0, 3, 1, 2)#10,512,512,5（channel）->#10,5,512,512,
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std

        surface_road = self.lgc_encoder(x)

        # [B, H, W, 2]
        surface_road= surface_road.permute(0, 2, 3, 1) #not use sigmoid,because of bce_loss  #10,1(c),512,512->10,512,512,1
        return surface_road

    def infer_masks_and_img_features(self, rgb):


        x = rgb.permute(0, 3, 1, 2)#64,512,512,5->64,5,512,512
        x = (x - self.pixel_mean) / self.pixel_std
        if self.config.ORIGINAL_DLINK_SAM_ROAD:  # original network
            image_embeddings, intermediate_features = self.image_encoder(x)#64,5(channel),512,512->64,256(channel),32,32
            surface_road = self.surface_road_net(x, intermediate_features)  # 64,1(channel),512,512
            surface_road = torch.sigmoid(surface_road)
            mask_logits = self.map_decoder(image_embeddings)  # 64,256(channel),32,32->64,2(channel),512,512
            mask_scores = torch.sigmoid(mask_logits)  # 64,2(channel),512,512
        else:
            surface_road= self.lgc_encoder(x)
            surface_road = torch.sigmoid(surface_road)  # use for reference.pys urface_road经过sigmoid（），所以数值在[0,1]

        # [B, H, W, 2]

        surface_road=surface_road.permute(0, 2, 3, 1)#64,512,512,1
        return surface_road


    def infer_toponet(self, image_embeddings, graph_points, pairs, valid):
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        return topo_scores

    def train_one_epoch(self, bce_loss_plt,dice_loss_plt,train_loss_plt,lear_rate,optimizer, data_loader, device, epoch):
        self.train()
        if epoch == self.config.TRAIN_EPOCHS - 1:
            save_dir = self.config.ROAD_RESULT_PATH
            if os.path.isdir(os.path.join(save_dir, "train")):
                shutil.rmtree(os.path.join(save_dir, "train"))
            os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)

        loss_train=AverageMeter()
        loss_bce=AverageMeter()
        loss_dice=AverageMeter()
        start=time.time()
        optimizer.zero_grad()
        train_data_loader = tqdm(data_loader, file=sys.stdout)
        train_data_loader.set_description(f'train epoch:[{epoch}/{self.config.TRAIN_EPOCHS}]')
        for step, batch in enumerate(train_data_loader):
            # masks: [B, H, W]
            rgb = batch['rgb']
            road_surface_label=batch['road_surface'].to(device)#完整道路面真值
            image_names=batch['image_name']
            # print(image_names)
            batch_size=rgb.size(0)
            surface_road_prediction = self(rgb.to(device))

            surface_loss=self.surface_criterion(surface_road_prediction,road_surface_label)
            if self.config.USE_BCE_DICE:
                surface_dice_loss= self.surface_dice(surface_road_prediction,road_surface_label)
                loss=surface_loss+surface_dice_loss
            else:
                loss = surface_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train.update(loss.item(),batch_size)
            loss_bce.update(surface_loss.item(),batch_size)
            if self.config.USE_BCE_DICE:
                loss_dice.update(surface_dice_loss.item(),batch_size)
        epoch_time =time.time()-start
        print(f'|time:{epoch_time:.2f}s'f'|train_loss:{loss_train.avg:.4f}'f'|bce_loss:{loss_bce.avg:.4f}'f'|dice_loss:{loss_dice.avg:.4f}')
        print(optimizer.param_groups[0]["lr"])
        train_loss_plt.append(loss_train.avg)
        bce_loss_plt.append(loss_bce.avg)
        dice_loss_plt.append(loss_dice.avg)
        lear_rate.append(optimizer.param_groups[0]["lr"])


        #print the road predict result
        return loss

    @torch.no_grad()
    def validation_one_epoch(self,precision1,recall1,f11,iou_p,optimizer,scheduler, data_loader, device, epoch):


        self.eval()
        iter_num = len(data_loader)
        epoch_metrics = 0
        if epoch == self.config.TRAIN_EPOCHS - 1:
            save_dir = self.config.ROAD_RESULT_PATH
            if os.path.isdir(os.path.join(save_dir, "test")):
                shutil.rmtree(os.path.join(save_dir, "test"))
            os.makedirs(os.path.join(save_dir, "test"), exist_ok=True)
        # masks: [B, H, W]
        val_data_loader = tqdm(data_loader, file=sys.stdout)
        val_data_loader.set_description(f'val epoch:[{epoch}/{self.config.TRAIN_EPOCHS}]')

        for step, batch in enumerate(val_data_loader):
            rgb= batch['rgb']
            road_surface_label=batch['road_surface'].to(device)#完整道路面真值
            image_names = batch['image_name']
            surface_road_prediction = self(rgb.to(device))
            surface_loss=self.surface_criterion(surface_road_prediction,road_surface_label)
            if self.config.USE_BCE_DICE:
                surface_dice_loss= self.surface_dice(surface_road_prediction,road_surface_label)
                loss=surface_loss+surface_dice_loss
            else:
                loss = surface_loss
            self.road_iou.update(surface_road_prediction, road_surface_label.to(device))
            self.topo_f1.update(surface_road_prediction, road_surface_label)
            metrics = self.metrics(road_surface_label, torch.sigmoid(surface_road_prediction))
            epoch_metrics += metrics

        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        epoch_metrics /= iter_num
        self.road_iou.reset()
        self.topo_f1.reset()
        if not os.path.exists(os.path.join(self.config.TRAIN_WEIGHT)):
            os.makedirs(os.path.join(self.config.TRAIN_WEIGHT))
        if (epoch + 1) % 10 == 0:
            checkpoint = {"epoch": epoch,
                          "state_dict": self.state_dict(),
                          "optimizer": optimizer.state_dict(),#
                          "scheduler": scheduler.state_dict(), #
                          }

            torch.save(checkpoint,
                       os.path.join(self.config.TRAIN_WEIGHT, f"epoch-{epoch}-f1{topo_f1:.2f}.ckpt"))

        return loss
    def configure_optimizers(self):
        param_dicts = []
        cen_encoder_params = {
            'params': [p for k, p in self.lgc_encoder.cen_encoder.named_parameters() if
                       'lgc_encoder.cen_encoder.' + k in self.matched_param_names],
            'lr': 0.00005,  # 0.00005
        }
        param_dicts.append(cen_encoder_params)

        surface_road_net_params = {
            'params': [p for p in self.lgc_encoder.curfnet.parameters()],
            'lr': 0.0002  # 0.0002
        }
        param_dicts.append(surface_road_net_params)

        if self.config.USE_LGFF:
            LGFF_net_params= {
                    'params': [p for p in self.lgc_encoder.LGFF.parameters()],
                    'lr': 0.0002# 0.0002
                }
            param_dicts.append(LGFF_net_params)
        if self.config.USE_SCATT or self.config.USE_SCMA:
            spatial_channel_params= {
                    'params': [p for p in self.lgc_encoder.curf_center_bridge.parameters()],
                    'lr': 0.0002#0.0002
                }
            param_dicts.append(spatial_channel_params)
        if self.config.USE_VSSM_SS2D :
            vssm_params= {
                    'params': [p for p in self.lgc_encoder.vssm_module.parameters()],
                    'lr': 0.0002# 0.0002
                }
            param_dicts.append(vssm_params)

        decoder_params = [{
            'params': [p for p in self.lgc_encoder.map_decoder.parameters()],
            'lr': 0.001#0.001
        }]
        param_dicts += decoder_params

        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,], gamma=0.1)
        return optimizer,step_lr

