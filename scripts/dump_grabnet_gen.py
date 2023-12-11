import argparse
import hashlib
import os
import pickle
from argparse import Namespace
from time import time
from typing import List

import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import lib.models
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.recorder import Recorder


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    dump_path = os.path.join(recorder.dump_path, "results")
    os.makedirs(dump_path, exist_ok=True)
    logger.info(f"Dumping results to {dump_path}")

    def dumper(prd, inp, step_idx, **kwargs):
        obj_id: List[str] = inp[Queries.OBJ_ID]
        obj_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
        obj_rotmat = inp[Queries.OBJ_ROTMAT].detach().cpu().numpy()
        if Queries.ALT_HAND_TRANSL_OBJ in inp:
            alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

        hand_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
        hand_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
        n_samples = hand_verts_c.shape[0]  # batch
        for i in range(n_samples):
            # calcluate the md5 of hand_verts_r, and only use the first 10 hashcode
            grasp_hash = hashlib.md5(hand_verts_r[i].tobytes()).hexdigest()[:10]
            sample_fname = f"{obj_id[i]}_{grasp_hash}.pkl"
            sample_content = {
                "obj_id": obj_id[i],
                "obj_rotmat": obj_rotmat[i],
                "obj_verts_ds": obj_verts_ds[i],
                "hand_verts_c": hand_verts_c[i],
                "hand_verts_r": hand_verts_r[i],
            }
            if Queries.ALT_HAND_TRANSL_OBJ in inp:
                sample_content["alt_hand_verts"] = alt_hand_verts[i]
            with open(os.path.join(dump_path, sample_fname), "wb") as f:
                pickle.dump(sample_content, f)
        return True

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=True,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test", callback=dumper)  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent, callback=dumper)  # IntGen


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify this variable when using the IntGen model.")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
