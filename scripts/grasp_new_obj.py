import argparse
import os
import time
from argparse import Namespace

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ready_arguments
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.nn.parallel import DataParallel as DP
from trimesh import Trimesh

from lib.datasets.grasp_query import Queries
from lib.models import *
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN
from lib.utils.logger import logger
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext

GrabNetConfig = dict(
    DATA_PRESET=dict(
        CENTER_IDX=9,
        N_RESAMPLED_OBJ_POINTS=4096,
    ),
    MODEL=dict(
        TYPE='GrabNet',
        COARSE_NET=dict(
            TYPE='CoarseNet',
            LATENTD=16,
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/CoarseNet.pth.tar',
        ),
        REFINE_NET=dict(
            TYPE='RefineNet',
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/RefineNet.pth.tar',
        ),
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformObject",
        RAND_ROT=False,
        USE_ORIGINAL_OBJ_ROT=True,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
    ),
)


def load_obj_models(obj_path: str, n_sample_verts=10000, rescale=False):
    obj_trimesh: Trimesh = trimesh.load(obj_path, process=False)
    obj_verts = np.asarray(obj_trimesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(obj_trimesh.faces, dtype=np.int32)

    maximum = obj_verts.max(0, keepdims=True)
    minimum = obj_verts.min(0, keepdims=True)
    offset = (maximum + minimum) / 2
    obj_verts = obj_verts - offset

    if rescale:
        # if rescale is true, we need to rescale the object to fit in radius=0.1m sphere
        scale = (obj_verts.max() - obj_verts.min()) / 2
        obj_verts = obj_verts / scale
        obj_verts = obj_verts * 0.1

    obj_rotmat = np.eye(3, dtype=np.float32)

    mesh = Meshes(verts=torch.from_numpy(obj_verts).unsqueeze(0), faces=torch.from_numpy(obj_faces).unsqueeze(0))
    obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, n_sample_verts, return_normals=True)

    res = {
        Queries.SAMPLE_IDENTIFIER: obj_path,
        Queries.OBJ_ID: obj_path,
        Queries.OBJ_VERTS_OBJ: torch.from_numpy(obj_verts),
        Queries.OBJ_FACES: torch.from_numpy(obj_faces),
        Queries.OBJ_ROTMAT: torch.from_numpy(obj_rotmat),
        Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds.squeeze(0),
        Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds.squeeze(0),
    }

    return res


def grasp_new_obj(arg: Namespace, exp_time):
    rank = 0
    cfg = CN(GrabNetConfig)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    model = DP(model).to(device=rank)
    transform = DP(transform).to(device=rank)
    model.eval()

    obj_data = load_obj_models(arg.obj_path, rescale=arg.rescale)
    for k, v in obj_data.items():
        if isinstance(v, torch.Tensor):
            obj_data[k] = v.unsqueeze(0).to(rank)
        if isinstance(v, np.ndarray):
            obj_data[k] = torch.from_numpy(v).unsqueeze(0).to(rank)

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    mano_rhand_path = os.path.join(arg.mano_path, "models", "MANO_RIGHT.pkl")
    mano_data = ready_arguments(mano_rhand_path)
    hand_faces = np.array(mano_data["f"]).astype(np.int32)

    res_hand_verts = []
    print("Press D to show next")
    for i in range(arg.n_grasps):
        show_next = False
        obj_data = transform(obj_data)
        prd, _ = model(inp=obj_data, step_idx=0, mode="test")

        hand_verts = prd["Refine.hand_verts"][0].detach().cpu().numpy()
        res_hand_verts.append(hand_verts)
        obj_verts = obj_data[Queries.OBJ_VERTS_OBJ][0].detach().cpu().numpy()
        obj_faces = obj_data[Queries.OBJ_FACES][0].detach().cpu().numpy()

        viz_ctx.update_by_mesh(f"hand", hand_verts, hand_faces, vcolors=CMap["tomato"])
        viz_ctx.update_by_mesh("obj", obj_verts, obj_faces, vcolors=CMap["lime"])

        while not show_next:
            viz_ctx.step()

    viz_ctx.deinit()

    if arg.save:
        ts = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(exp_time))
        save_path = os.path.join("user", "grasps", ts)
        os.makedirs(save_path, exist_ok=True)
        for i, hand_verts in enumerate(res_hand_verts):
            hand_mesh = Trimesh(vertices=hand_verts, faces=hand_faces)
            hand_mesh.export(os.path.join(save_path, f"hand_{i}.obj"))
        obj_mesh = Trimesh(vertices=obj_verts, faces=obj_faces)
        obj_mesh.export(os.path.join(save_path, "object.obj"))


if __name__ == '__main__':
    exp_time = time.time()
    arg, _ = parse_exp_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--obj_path", type=str, required=True, help='The path to the 3D object Mesh')
    parser.add_argument("--mano_path", type=str, default="assets/mano_v1_2", help='The path to MANO models')
    parser.add_argument("--n_grasps", type=int, required=False, default=1, help='how many grasps to generate')
    parser.add_argument("--rescale",
                        action="store_true",
                        default=False,
                        help='rescale the object to fit in radius=0.1m sphere')
    parser.add_argument("--save", action="store_true", default=False, help='save the grasps to file')
    arg_extra, _ = parser.parse_known_args()
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    grasp_new_obj(arg, exp_time)
