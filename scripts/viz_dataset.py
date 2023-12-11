import argparse

import numpy as np
from torch.utils.data import DataLoader

from lib.datasets import create_dataset
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.models import *
from lib.utils import builder
from lib.utils.config import CN
from lib.utils.net_utils import setup_seed
from lib.viztools.utils import ColorsMap as Cmap
from lib.viztools.utils import get_color_map
from lib.viztools.viz_o3d_utils import VizContext

CFG_OIShape = dict(
    DATASET=dict(
        TYPE="OIShape",
        DATA_ROOT="./data",
        DATA_SPLIT=None,
        OBJ_CATES="all",
        DATA_MODE="grasp",
        INTENT_MODE=["use", "hold", "liftup"],
    ),
    DATA_PRESET=dict(
        CENTER_IDX=9,
        USE_CACHE=True,
        N_RESAMPLED_OBJ_POINTS=4096,
        N_INTENTS=3,
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformHandObject",
        RAND_ROT=True,
        RAND_DEG_STD=20,
        USE_ORIGINAL_OBJ_ROT=False,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
        POSE_DISTURBER=dict(
            tsl_sigma=0.02,
            pose_sigma=0.2,
            root_rot_sigma=0.004,
        ),
    ),
)

CFG_OIShapeHandOver = dict(
    DATASET=dict(
        TYPE="OIShape",
        DATA_ROOT="./data",
        DATA_SPLIT=None,
        OBJ_CATES="all",
        INTENT_MODE=["handover"],
        DATA_MODE="handover",
    ),
    DATA_PRESET=dict(
        CENTER_IDX=9,
        USE_CACHE=True,
        N_RESAMPLED_OBJ_POINTS=4096,
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformHandover",
        RAND_ROT=True,
        RAND_DEG_STD=20,
        USE_ORIGINAL_OBJ_ROT=False,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
        POSE_DISTURBER=dict(
            tsl_sigma=0.02,
            pose_sigma=0.2,
            root_rot_sigma=0.004,
        ),
    ),
)

CFG_GrabNetData = dict(
    DATASET=dict(
        TYPE="GrabNetData",
        DATA_ROOT="./data",
        DATA_SPLIT=None,
    ),
    DATA_PRESET=dict(
        CENTER_IDX=9,
        USE_CACHE=True,
        N_RESAMPLED_OBJ_POINTS=4096,
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformHandObject",
        RAND_ROT=True,
        RAND_DEG_STD=20,
        USE_ORIGINAL_OBJ_ROT=False,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
        POSE_DISTURBER=dict(
            tsl_sigma=0.02,
            pose_sigma=0.2,
            root_rot_sigma=0.004,
        ),
    ),
)
config_factory = {
    "oishape": CFG_OIShape,
    "oishape_handover": CFG_OIShapeHandOver,
    "grabnet": CFG_GrabNetData,
}


def get_config(dname):
    assert dname in config_factory.keys(), f"Dataset {dname} not supported!"
    return config_factory[dname]


def main(args):
    setup_seed(seed=0)
    cfg = CN(get_config(args.dataset))
    if args.dataset != "mix":
        cfg.DATASET.DATA_SPLIT = args.data_split
    else:
        for _, el in enumerate(cfg.DATASET.CONTENT):
            el.DATA_SPLIT = args.data_split

    dataset = create_dataset(cfg.DATASET, data_preset=cfg.DATA_PRESET)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=0,
                             drop_last=False,
                             collate_fn=grasp_data_collate)
    data_iter = iter(data_loader)

    viz_context = VizContext()
    viz_context.init(point_size=10.0)

    def next_sample(_):
        grasp_item = next(data_iter)
        grasp_item = transform(grasp_item)
        sample_id = grasp_item[Queries.SAMPLE_IDENTIFIER]
        print(sample_id[0])

        if Queries.HAND_VERTS_OBJ in grasp_item:
            hand_verts = grasp_item[Queries.HAND_VERTS_OBJ].squeeze(0).numpy()
            hand_faces = grasp_item[Queries.HAND_FACES].squeeze(0).numpy()
            hand_verts_f = grasp_item[f"{Queries.HAND_VERTS_OBJ}_f"].squeeze(0).numpy()
            viz_context.update_by_mesh("hand_f", hand_verts_f, hand_faces, vcolors=Cmap["deepskyblue"], update=True)
            viz_context.update_by_mesh("hand", hand_verts, hand_faces, vcolors=Cmap["tomato"], update=True)

        if Queries.ALT_HAND_VERTS_OBJ in grasp_item:
            alt_hand_verts = grasp_item[Queries.ALT_HAND_VERTS_OBJ].squeeze(0).numpy()
            viz_context.update_by_mesh("alt_hand", alt_hand_verts, hand_faces, vcolors=Cmap["blue"], update=True)

        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS].squeeze(0).numpy()
        obj_normals_ds = grasp_item[Queries.OBJ_NORMALS_OBJ_DS].squeeze(0).numpy()
        viz_context.update_by_pc("obj", obj_verts_ds, obj_normals_ds, pcolors=Cmap["lime"], update=True)

        if Queries.OBJ_BPS in grasp_item:  # find a way to check the bps
            basis_points = transform.bps_basis
            obj_bps = grasp_item[Queries.OBJ_BPS].squeeze(0).numpy()
            obj_bps = obj_bps / np.linalg.norm(basis_points, axis=-1).max()  # norm it to [0, 1]
            # get all the basis_points with z value < 0, for better viz
            viz_idx = np.where(basis_points[:, -1] < 0)[0]
            basis_points = basis_points[viz_idx]
            obj_bps = obj_bps[viz_idx]
            obj_bps_for_viz = 1 - obj_bps  # inv the min-max, so that the red and blue are swapped
            bps_color = get_color_map(obj_bps_for_viz, "contactness")
            viz_context.update_by_pc("bps", basis_points, pcolors=bps_color, update=True)

    next_sample(viz_context)
    viz_context.register_key_callback("D", next_sample)
    viz_context.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="viz grabnet grasp")
    parser.add_argument('-d',
                        "--dataset",
                        type=str,
                        default="oishape",
                        help="dataset name",
                        choices=["oishape", "oishape_handover", "grabnet"])
    parser.add_argument('-sp',
                        "--data_split",
                        type=str,
                        default="train",
                        choices=["train", "test", "val", "all"],
                        help="data split")

    args = parser.parse_args()
    main(args)
