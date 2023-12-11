import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
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
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

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

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    logger.info("You are viz GrabNet's prediciton")
    model.eval()
    from lib.utils.pcd import point2point_signed
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen

            h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
            h_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
            h_faces = mano_layer.th_faces.detach().cpu().numpy()
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
            o_norms_ds = inp[Queries.OBJ_NORMALS_OBJ_DS].detach().cpu().numpy()
            if dataset.data_mode == "handover":
                alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

            n_items = h_verts_c.shape[0]
            viz_ofs = np.array([0.2, 0.0, 0.0])
            for i in range(n_items):
                show_next = False
                viz_ctx.update_by_mesh("hand_coarse", h_verts_c[i], h_faces, vcolors=CMap["deepskyblue"])
                viz_ctx.update_by_mesh("hand_refine", h_verts_r[i] + viz_ofs, h_faces, vcolors=CMap["tomato"])
                ov = o_verts_ds[i]
                on = o_norms_ds[i]
                viz_ctx.update_by_pc("obj", ov, on, pcolors=CMap["lime"])
                viz_ctx.update_by_pc("obj_ofs", ov + viz_ofs, on, pcolors=CMap["lime"])

                if dataset.data_mode == "handover":
                    viz_ctx.update_by_mesh("alt_hand", alt_hand_verts[i], h_faces, vcolors=CMap["blue"])
                    viz_ctx.update_by_mesh("alt_hand_ofs", alt_hand_verts[i] + viz_ofs, h_faces, vcolors=CMap["blue"])

                while not show_next:
                    viz_ctx.step()


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
                        help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
