import argparse
import logging
import os
from datetime import datetime
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import mmcv
from model.builder import build_model
from third_party.unimatch.supervised import evaluate2
from third_party.unimatch.dataset.semicd import SemiCDDataset
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import init_log


def load_checkpoint(model, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")
    return missing_keys, unexpected_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--port', default=12345, type=int, help='Master port for distributed')
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed')
    args = parser.parse_args()

    # 1. 读取配置
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # 2. 初始化日志
    logger = init_log('global', logging.INFO)
    logger.propagate = False
    mmcv.utils.get_logger('mmcv').setLevel('WARNING')

    # 3. 设置分布式环境变量，单卡时必需
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 4. 初始化分布式环境（单卡）
    rank, world_size = setup_distributed(port=args.port)

    # 5. 设置 CUDA 设备
    torch.cuda.set_device(args.local_rank)

    # 6. 构建模型并转为 SyncBatchNorm
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)

    # 7. **单卡不包 DDP，注释掉下面代码**
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.local_rank], output_device=args.local_rank,
    #     broadcast_buffers=False, find_unused_parameters=True
    # )

    # 8. 加载 checkpoint，去除 'module.' 前缀
    if os.path.isfile(args.checkpoint):
        load_checkpoint(model, args.checkpoint, logger)
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}")

    # 9. 准备验证集和分布式采样器
    valset = SemiCDDataset(cfg, 'val')
    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        valset, batch_size=1, shuffle=False,
        sampler=val_sampler, num_workers=2, pin_memory=True
    )

    cudnn.enabled = True
    cudnn.benchmark = True

    # 10. 评估
    eval_mode = cfg.get('eval_mode', 'default_eval_mode')
    results = evaluate(model, val_loader, eval_mode, cfg, return_cd_metric=True)

    # 11. 只在 rank0 打印结果
    if rank == 0:
        mIoU, iou_class, overall_acc, f1_class, precision_class, recall_class = results
        logger.info('***** Evaluation Results *****')
        logger.info(f'Mean IoU: {mIoU:.4f}')
        logger.info(f'IoU (Unchanged/Changed): {iou_class[0]:.4f} / {iou_class[1]:.4f}')
        logger.info(f'F1 (Unchanged/Changed): {f1_class[0]:.4f} / {f1_class[1]:.4f}')
        logger.info(f'Precision (Unchanged/Changed): {precision_class[0]:.4f} / {precision_class[1]:.4f}')
        logger.info(f'Recall (Unchanged/Changed): {recall_class[0]:.4f} / {recall_class[1]:.4f}')
        logger.info(f'Overall Accuracy: {overall_acc:.4f}')

    # 12. 关闭分布式进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
#python eval.py --config configs/eval_config.yaml --checkpoint exp/exp-48/best.pth
