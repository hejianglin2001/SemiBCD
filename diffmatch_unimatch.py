# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import pprint
import shutil
import uuid
import time
from datetime import datetime

import mmcv
import torch
import torch.backends.cudnn as cudnn
import yaml
import matplotlib

matplotlib.use('Agg')  # 设置使用 Agg 后端

from matplotlib import pyplot as plt
from mmseg.core import build_optimizer
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.palettes import get_palette
from experiments import get_git_revision
from model.builder import build_model
from model.utils import BCLLoss
from third_party.unimatch.supervised import evaluate
from third_party.unimatch.dataset.semicd import SemiCDDataset
from datasets.classes import CLASSES
from third_party.unimatch.util.ohem import ProbOhemCrossEntropy2d
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import count_params, count_training_params, init_log
from utils.gen_code_archive import gen_code_archive
from utils.plot_utils import plot_data, colorize_label
from utils.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask)
from version import __version__

import torch.nn.functional as F



def compute_lsc_loss(predA, predB, vl_mask,
                     m_u=0.985, m_c=0.95, alpha=5.0):
    """
    Margin-hinge + balanced weighting LSC loss.
    - predA, predB: [B_2, C, H, W] 分割 logits
    - vl_mask:      [B_2, H, W], 0=unchanged, 1=changed, 255=ignore
    - m_u: 不变区希望 cos >= m_u；只对 cos < m_u 部分惩罚
    - m_c: 变化区希望 cos <= m_c；只对 cos > m_c 部分惩罚
    - alpha: 对“变化”区域再额外放大梯度
    """
    # 1. 构造三类掩码
    valid = (vl_mask != 255)
    unchanged = (vl_mask == 0) & valid
    changed = (vl_mask == 1) & valid

    # 2. 计算 cos_sim
    pA = F.normalize(F.softmax(predA, dim=1), p=2, dim=1)
    pB = F.normalize(F.softmax(predB, dim=1), p=2, dim=1)
    cos_sim = torch.sum(pA * pB, dim=1)  # [B_2,H,W]

    # 3. margin-hinge
    loss_u = F.relu(m_u - cos_sim) * unchanged.float()
    loss_c = F.relu(cos_sim - m_c) * changed.float()

    # 4. 分别归一化并加权
    nu = unchanged.sum().clamp_min(1.0)
    nc = changed.sum().clamp_min(1.0)
    loss = loss_u.sum() / nu + alpha * loss_c.sum() / nc

    # 最后再平均
    return loss / 2.0
def compute_lfc_loss(feat1, feat2, change_mask, margin=1.0, alpha=5.0):
    """
    feat1, feat2: [B_2, C, H, W] 特征输出（建议做过 F.normalize）
    change_mask: [B_2, H, W]，0 表示未变，1 表示有变化，255 忽略
    """


    if feat1.shape[-2:] != change_mask.shape[-2:]:
        change_mask = F.interpolate(
            change_mask.unsqueeze(1).float(),
            size=feat1.shape[-2:],
            mode='nearest'  # 因为是离散标签
        ).squeeze(1).long()
    # 归一化使得相似度度量稳定
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)

    # 计算特征距离（L2 距离的平方 = 1 - cos + 1 - cos）
    dist = ((feat1 - feat2)**2).sum(dim=1)  # [B_2, H, W]

    valid = (change_mask != 255)
    unchanged = (change_mask == 0) & valid
    changed = (change_mask == 1) & valid
    #print(f"dist shape: {dist.shape}, unchanged shape: {unchanged.shape}")
    # 对于未变化区域，直接用距离作为损失
    loss_u = dist * unchanged.float()
    # 对于变化区域，希望特征差异至少大于 margin
    loss_c = F.relu(margin - dist) * changed.float()

    nu = unchanged.sum().clamp_min(1.0)
    nc = changed.sum().clamp_min(1.0)

    return (loss_u.sum() / nu + alpha * loss_c.sum() / nc) / 2.0




def compute_cross_ce_with_mask(pred1, pred2, tau=0.70,mask=None):
    with torch.no_grad():
        pred1_prob = F.softmax(pred1, dim=1)
        if mask is None:
            max_probs, _ = pred1_prob.max(dim=1)
            mask = (max_probs >= tau).float()
    log_pred2 = F.log_softmax(pred2, dim=1)
    loss = -(pred1_prob * log_pred2).sum(dim=1) * mask
    return loss.mean()

def seed_everything(seed):
    import random, os
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    seed_everything(cfg.get('seed', 42))
    labeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/labeled.txt'
    unlabeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/unlabeled.txt'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    mmcv.utils.get_logger('mmcv').setLevel('WARNING')
    # rank = 0
    # world_size = 1
    rank, world_size = setup_distributed(port=args.port)
    if cfg['nccl_p2p_disable']:
        os.environ["NCCL_P2P_DISABLE"] = str(1)

    if rank == 0:
        timestr = datetime.now().strftime("%y%m%d-%H%M")
        uid = str(uuid.uuid4())[:5]
        run_name = f'{timestr}_{cfg["name"]}_v{__version__}_{uid}'.replace('.', '-')
        save_path = f'exp/exp-{cfg["exp"]}/{run_name}'
        os.makedirs(save_path, exist_ok=True)

        formatter = logging.Formatter(fmt='[%(asctime)s] [%(levelname)-8s] %(message)s')
        fileHandler = logging.FileHandler(f'{save_path}/debug.log')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        all_args = {**cfg, **vars(args),
                    'labeled_id_path': labeled_id_path, 'unlabeled_id_path': unlabeled_id_path,
                    'ngpus': world_size, 'run_name': run_name, 'save_path': save_path,
                    'exec_git_rev': get_git_revision(), 'exec_version': __version__}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(save_path)

        shutil.copyfile(args.config, os.path.join(save_path, 'config.yaml'))
        with open(os.path.join(save_path, 'all_args.yaml'), 'w') as f:
            yaml.dump(all_args, f, default_flow_style=None, sort_keys=False, indent=2)
        gen_code_archive(save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    vl_consistency_lambda = cfg['vl_consistency_lambda']
    vl_loss_reduce = cfg['vl_loss_reduce']
    assert vl_loss_reduce in ['mean', 'mean_valid', 'mean_all']
    assert cfg['use_fp']
    assert cfg['pleval']

    contrastive_loss_weight = cfg['contrastive_loss_weight']

    model = build_model(cfg)
    if 'optimizer' not in cfg:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(model, cfg['optimizer'])
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    if rank == 0:
        logger.info(model)
        logger.info(f'Total params: {count_params(model):.1f}M\n')
        if hasattr(model, 'backbone'):
            logger.info(
                f'Backbone params (training/total): {count_training_params(model.backbone):.1f}M/{count_params(model.backbone):.1f}M\n')
        if hasattr(model, 'decode_head'):
            logger.info(
                f'Decoder params (training/total): {count_training_params(model.decode_head):.1f}M/{count_params(model.decode_head):.1f}M\n')

    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = 0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'mmseg':
        criterion_l = None
    else:
        raise ValueError(cfg['criterion_u']['name'])

    criterion_dist = BCLLoss(margin=2.0, loss_weight=1.0, ignore_index=255)

    if cfg['criterion_u'] == 'CELoss':
        criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    elif cfg['criterion_u'] == 'mmseg':
        criterion_u = None
    else:
        raise ValueError(cfg['criterion_u'])

    if vl_consistency_lambda != 0:
        if vl_loss_reduce == 'mean':
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255).cuda()
        elif vl_loss_reduce in ['mean_valid', 'mean_all']:
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
        else:
            raise ValueError(vl_loss_reduce)
    unlabeled_id_path_mix = f'splits/whu/train.txt'
    labeled_id_path_mix = f'splits/levir/train.txt'
    trainset_u = SemiCDDataset(cfg, 'train_u', id_path=unlabeled_id_path,isMix=False)
    trainset_l = SemiCDDataset(cfg, 'train_l', id_path=labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(cfg, 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True,
                               sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True,
                               sampler=trainsampler_u)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    palette = get_palette(cfg['dataset'])

    if cfg['iters'] is not None:
        assert cfg['epochs'] is None
        cfg['epochs'] = math.ceil(cfg['iters'] / len(trainloader_u))

    total_iters = len(trainloader_u) * cfg['epochs']
    scheduler_max_iters = cfg.get('scheduler_max_iters', total_iters)
    assert scheduler_max_iters >= total_iters
    if rank == 0:
        logger.info(f'Train for {cfg["epochs"]} epochs / {total_iters} iterations.')
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1
    m_u, m_c = 0.985, 0.95
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info(
                '===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        len_loader = min(len(trainloader_l), len(trainloader_u))

        for i, ((imgA_x, imgB_x, mask_x, vl_mask_x_change, vl_mask_x_A, vl_mask_x_B),
                (imgA_w, imgB_w, imgA_s1, imgB_s1,
                 imgA_s2, imgB_s2, ignore_mask, mix1, _, vl_mask_change, vl_mask_A, vl_mask_B)) in enumerate(loader):


            t0 = time.time()
            iters = epoch * len(trainloader_u) + i
            imgA_x, imgB_x = imgA_x.cuda(), imgB_x.cuda()
            imgA_s1, imgB_s1 = imgA_s1.cuda(), imgB_s1.cuda()
            imgA_s2, imgB_s2 = imgA_s2.cuda(), imgB_s2.cuda()
            mask_x = mask_x.cuda()
            imgA_w, imgB_w = imgA_w.cuda(), imgB_w.cuda()


            # for VL pseudo labels
            vl_mask_x_change = vl_mask_x_change.cuda()
            vl_mask_x_A, vl_mask_x_B = vl_mask_x_A.cuda(), vl_mask_x_B.cuda()
            vl_mask_A, vl_mask_B = vl_mask_A.cuda(), vl_mask_B.cuda()

            vl_mask_change = vl_mask_change.cuda()


            model.train()

            pred_x, pred_x_for_vl, pred_x_segA, pred_x_segB, pred_x_dist, feat_x_A, feat_x_B = model(imgA_x, imgB_x,
                                                                                 need_seg_aux=True, need_contrast=True)

            pred_fp, pred_fp_for_vl, pred_fp_segA, pred_fp_segB, pred_fp_dist, feat_w_A, feat_w_B = model(imgA_w, imgB_w, need_seg_aux=True,
                                                                                      need_contrast=True, need_fp=True)
            pred_w = pred_fp[0]
            pred_w_fp = pred_fp[1]
            pred_w_vl = pred_fp_for_vl[0]
            pred_w_fp_vl = pred_fp_for_vl[1]
            pred_w_segA = pred_fp_segA[0]
            pred_w_segA_fp = pred_fp_segA[1]
            pred_w_segB = pred_fp_segB[0]
            pred_w_segB_fp = pred_fp_segB[1]
            pred_w_dist = pred_fp_dist[0]
            pred_w_dist_fp = pred_fp_dist[1]

            pred_w = pred_w.detach()

            pred_s, pred_s_for_vl, pred_s_segA, pred_s_segB, pred_s_dist, feat_s_A, feat_s_B = model(imgA_s1, imgB_s1, need_seg_aux=True,
                                                                                 need_contrast=True)


            # 有标签的变化检测监督
            # 变化检测
            loss_x = criterion_l(pred_x, mask_x)
            # 双时分割任务
            loss_x_segA = criterion_l(pred_x_segA, vl_mask_x_A)
            loss_x_segB = criterion_l(pred_x_segB, vl_mask_x_B)

            
            
            ramp_factor = min(1.0, iters / (0.8 * 200 * len_loader))
            
            ramp_factor = 1.0

           
            # Lp2w
            loss_w_cd = criterion_l(pred_w, vl_mask_change)
            loss_w_segA = criterion_l(pred_w_segA, vl_mask_A)
            loss_w_segB = criterion_l(pred_w_segB, vl_mask_B)
            lp2w = (loss_w_segA + loss_w_segB)/4.0 + loss_w_cd * 4.5

            # Lw2f
            lw2f_A = compute_cross_ce_with_mask(pred_w_segA.detach(), pred_w_segA_fp)
            lw2f_B = compute_cross_ce_with_mask(pred_w_segB.detach(), pred_w_segB_fp)
            lw2f_cd = compute_cross_ce_with_mask(pred_w.detach(), pred_w_fp)
            lw2f =  (lw2f_A + lw2f_B) * 1.5 + lw2f_cd * 6.0

            # Lw2i
            lw2i_A = compute_cross_ce_with_mask(pred_w_segA.detach(), pred_s_segA)
            lw2i_B = compute_cross_ce_with_mask(pred_w_segB.detach(), pred_s_segB)
            lw2i_cd = compute_cross_ce_with_mask(pred_w.detach(), pred_s)
            lw2i = (lw2i_A + lw2i_B) * 1.5 + lw2i_cd * 6.0
           
            # L_mvc
            loss_mvc = (lw2f + lw2i + lp2w)*ramp_factor
            

                # 计算 Lsc 损失
            lsc_s_cd = compute_lsc_loss(pred_s_segA, pred_s_segB, vl_mask_change, m_u=m_u, m_c=m_c)*ramp_factor
            lsc_w_cd = compute_lsc_loss(pred_w_segA, pred_w_segB, vl_mask_change, m_u=m_u, m_c=m_c)*ramp_factor
            lsc_x_cd = compute_lsc_loss(pred_x_segA, pred_x_segB, mask_x, m_u=m_u, m_c=m_c)
            loss_lsc = (lsc_s_cd + lsc_w_cd + lsc_x_cd * 1.5) / 3.0
            # L_fc
            lfc_x = compute_lfc_loss(feat_x_A, feat_x_B, mask_x)
            lfc_w = compute_lfc_loss(feat_w_A, feat_w_B, vl_mask_change)*ramp_factor
            lfc_s = compute_lfc_loss(feat_s_A, feat_s_B, vl_mask_change)*ramp_factor
            loss_lfc = (lfc_x * 1.5 + lfc_w + lfc_s) / 3.0
            # L_tha
            l_tha = (loss_lfc + loss_lsc) / 2.0


            loss_supervised = (loss_x_segA + loss_x_segB ) / 4.0


            loss = (loss_x * 2.0)

            loss = loss + (
                  l_tha  +
                   loss_mvc +
                    loss_supervised  ) / 3.0


            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'optimizer' not in cfg:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    lr = cfg['lr'] * (1 - k)
                else:
                    lr = cfg['lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            else:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - k)
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - iters / scheduler_max_iters) ** 0.9

            # Logging
            log_avg.update({
                'train/iter_time': time.time() - t0,
                'train/loss_all': loss,
                'train/loss_x': loss_x,
                'train/loss_w_cd': loss_w_cd,
               
                'train/loss_mvc': loss_mvc,
                'train/loss_tha': l_tha,
                'train/losssc': loss_lsc,
                'train/lsc_x_cd': lsc_x_cd,
                'train/lsc_w_cd': lsc_w_cd,
                'train/lsc_s_cd': lsc_s_cd,
                'train/loss_lfc': loss_lfc,
                'train/lfc_x_cd': lfc_x,
                'train/lfc_w_cd': lfc_w,
                'train/lfc_s_cd': lfc_s,
                'train/loss_w_seg': loss_w_segA + loss_w_segB,
                'train/loss_x_seg': loss_x_segA + loss_x_segB,
               
                'train/lp2w': lp2w,
                'train/lw2f': lw2f,
                'train/lw2i': lw2i,
            })



            if (i % 100 == 0
                    and rank == 0
            ):
                logger.info(f'Iters: {i} ' + str(log_avg))
                for k, v in log_avg.avgs.items():
                    writer.add_scalar(k, v, iters)

                log_avg.reset()

            if iters % len(trainloader_u) == 0 and rank == 0:
                print('Save debug images at iteration', iters)
                out_dir = os.path.join(save_path, 'debug')
                os.makedirs(out_dir, exist_ok=True)
                for b_i in range(imgA_x.shape[0]):
                    rows, cols = 9, 4
                    plot_dicts = [
                        dict(title='ImageA L', data=imgA_x[b_i], type='image'),
                        dict(title='ImageA w', data=imgA_w[b_i], type='image'),
                        dict(title='ImageA S1', data=imgA_s1[b_i], type='image'),
                        dict(title='ImageA S2', data=imgA_s2[b_i], type='image'),

                        dict(title='ImageB L', data=imgB_x[b_i], type='image'),
                        dict(title='ImageB w', data=imgB_w[b_i], type='image'),
                        dict(title='ImageB S1', data=imgB_s1[b_i], type='image'),
                        dict(title='ImageB S2', data=imgB_s2[b_i], type='image'),

                        dict(title='ImageA L, VL_mask', data=vl_mask_x_A[b_i], type='label', palette=palette),
                        dict(title='ImageA w, VL_mask', data=vl_mask_A[b_i], type='label', palette=palette),
                        dict(title='ImageA S1, VL_mask', data=vl_mask_A[b_i], type='label', palette=palette),
                        dict(title='ImageA S2, VL_mask', data=vl_mask_A[b_i], type='label', palette=palette),

                        dict(title='ImageB L, VL_mask', data=vl_mask_x_B[b_i], type='label', palette=palette),
                        dict(title='ImageB w, VL_mask', data=vl_mask_B[b_i], type='label', palette=palette),
                        dict(title='ImageB S1, VL_mask', data=vl_mask_B[b_i], type='label', palette=palette),
                        dict(title='ImageB S2, VL_mask', data=vl_mask_B[b_i], type='label', palette=palette),

                        dict(title='L_CD, GT', data=mask_x[b_i], type='label', palette=palette),
                        dict(title='w_CD, VL_mask', data=vl_mask_change[b_i], type='label', palette=palette),
                        dict(title='S1_CD, VL_mask', data=vl_mask_change[b_i], type='label', palette=palette),
                        dict(title='S2_CD, VL_mask', data=vl_mask_change[b_i], type='label', palette=palette),

                        dict(title='Pred L segA', data=pred_x_segA[b_i], type='prediction', palette=palette),
                        dict(title='Pred w segA', data=pred_w_segA[b_i], type='prediction', palette=palette),
                        dict(title='Pred S1 segA', data=pred_s_segA[b_i], type='prediction', palette=palette),

                        None,

                        dict(title='Pred L segB', data=pred_x_segB[b_i], type='prediction', palette=palette),
                        dict(title='Pred w segB', data=pred_w_segB[b_i], type='prediction', palette=palette),
                        dict(title='Pred S1 segB', data=pred_s_segB[b_i], type='prediction', palette=palette),

                        None,

                        dict(title='Pred L', data=pred_x[b_i], type='prediction', palette=palette),
                        dict(title='Pred w', data=pred_w[b_i], type='prediction', palette=palette),
                        dict(title='Pred S1', data=pred_s[b_i], type='prediction', palette=palette),

                        None,

                        None,
                        dict(title='PreCD w FP', data=pred_w_fp[b_i], type='prediction', palette=palette),
                        dict(title='PreA w FP', data=pred_w_segA_fp[b_i], type='prediction', palette=palette),
                        dict(title='PreB w FP', data=pred_w_segB_fp[b_i], type='prediction', palette=palette),
                    ]

                    fig, axs = plt.subplots(
                        rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0})
                    for ax, plot_dict in zip(axs.flat, plot_dicts):
                        if plot_dict is not None:
                            plot_data(ax, **plot_dict)
                    plt.savefig(os.path.join(out_dir, f'{(iters):07d}_{rank}-{b_i}.png'))
                    plt.close()
        if epoch % cfg.get('eval_every_n_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            eval_mode = cfg['eval_mode']
            mIoU, iou_class, overall_acc, f1_class, precision_class, recall_class = evaluate(model, valloader,
                                                                                             eval_mode, cfg,
                                                                                             return_cd_metric=True)

            if rank == 0:
                logger.info(run_name)
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))
                logger.info('***** Evaluation ***** >>>> IoU (Unchanged/Changed): {:.2f}/{:.2f}'.format(iou_class[0],
                                                                                                        iou_class[1]))
                logger.info('***** Evaluation ***** >>>> F1 (Unchanged/Changed): {:.2f}/{:.2f}'.format(f1_class[0],
                                                                                                       f1_class[1]))
                logger.info('***** Evaluation ***** >>>> Precision (Unchanged/Changed): {:.2f}/{:.2f}'.format(
                    precision_class[0], precision_class[1]))
                logger.info(
                    '***** Evaluation ***** >>>> Recall (Unchanged/Changed): {:.2f}/{:.2f}'.format(recall_class[0],
                                                                                                   recall_class[1]))
                logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))

                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = iou_class[1] > previous_best_iou
            previous_best_iou = max(iou_class[1], previous_best_iou)
            if is_best:
                previous_best_acc = overall_acc

            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                if is_best:
                    torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
