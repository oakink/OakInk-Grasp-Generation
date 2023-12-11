import os
from argparse import Namespace
from time import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.datasets import create_dataset
from lib.datasets.grasp_data import grasp_data_collate
from lib.models.model_abc import ModelABC as ModelABC
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix, format_args_cfg
from lib.utils.net_utils import (build_optimizer, build_scheduler, clip_gradient, setup_seed, worker_init_fn)
from lib.utils.recorder import Recorder
from lib.utils.summarizer import DDPSummaryWriter


def setup_ddp(arg, rank, world_size):
    """Setup distributed data parallel

    Args:
        arg (Namespace): arguments
        rank (int): rank of current process
        world_size (int): total number of processes, equal to number of GPUs
    """
    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert rank == torch.distributed.get_rank(), "Something wrong with DDP setup"
    torch.cuda.set_device(rank)
    dist.barrier()


def main_worker(rank: int, cfg: CN, arg: Namespace, world_size, time_f: float):
    setup_ddp(arg, rank, world_size)
    setup_seed(rank + cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f, root_path="exp")
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)

    dist.barrier()  # wait for recoder to finish setup
    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=int(arg.workers),
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              worker_init_fn=worker_init_fn,
                              collate_fn=grasp_data_collate,
                              persistent_workers=(int(arg.workers) > 0))

    if rank == 0:
        val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
        val_loader = DataLoader(val_data,
                                batch_size=arg.val_batch_size,
                                shuffle=True,
                                num_workers=int(arg.workers),
                                drop_last=False,
                                worker_init_fn=worker_init_fn,
                                collate_fn=grasp_data_collate)
    else:
        val_loader = None

    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    model.to(rank)
    transform.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS)
    transform = DDP(transform, device_ids=[rank], find_unused_parameters=False)

    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
    epoch = 0
    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)

    dist.barrier()  # wait for all processes to finish loading model
    n_steps = len(train_loader) * cfg.TRAIN.EPOCH
    logger.warning(f"===== start training [{epoch}, {cfg.TRAIN.EPOCH}), total iters: {n_steps} >>>>")
    for epoch_idx in range(epoch, cfg.TRAIN.EPOCH):
        train_sampler.set_epoch(epoch_idx)

        model.train()
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            batch = transform(batch)
            prd, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
            loss = loss_dict["loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()

            trainbar.set_description(f"{bar_prefix['train']} Epoch {epoch_idx} | {loss.item():.4f}")

        scheduler.step()
        dist.barrier()  # wait for all processes to finish training
        logger.info(f"{bar_prefix['train']} Epoch {epoch_idx} | loss: {loss.item():.4f}, Done")
        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        model.module.on_train_finished(recorder, epoch_idx)

        if (rank == 0  # only at rank 0,
                and epoch_idx != cfg.TRAIN.EPOCH - 1  # not the last epoch
                and epoch_idx % arg.eval_freq == 0):  # at eval freq, do validation
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=rank)
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * len(val_loader) + bidx
                    batch = transform(batch)
                    prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

            model.module.on_val_finished(recorder, epoch_idx)

    dist.destroy_process_group()
    # do last evaluation
    if rank == 0:
        logger.info("do last validation and save results")
        with torch.no_grad():
            model.eval()
            valbar = etqdm(val_loader, rank=rank)
            for bidx, batch in enumerate(valbar):
                step_idx = epoch_idx * len(val_loader) + bidx
                batch = transform(batch)
                prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

        model.module.on_val_finished(recorder, epoch_idx)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Use Distributed Data Parallel <====")
    mp.spawn(main_worker, args=(cfg, arg, world_size, exp_time), nprocs=world_size)
