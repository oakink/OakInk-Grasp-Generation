import os
from collections import namedtuple

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from oikit.oi_shape import OakInkShape
from oikit.oi_shape.utils import ALL_INTENT
from yacs.config import CfgNode as CN

from lib.datasets.grasp_data import GraspData
from lib.datasets.grasp_query import Queries
from lib.utils.builder import DATASET
from lib.viztools.viz_o3d_utils import VizContext

OIShapeGrasp = namedtuple(
    "OIShapeGrasp",
    [
        "split",
        "sample_idx",
        "obj_id",
        "contact_region",
        "contactness",
        "obj_verts_obj_processed",
        "obj_rot",
        "obj_transl",
        "sbj_name",
        "hand_verts_obj",
        "joints_obj",
        "hand_pose_obj",
        "hand_shape",
    ],
)
OIShapeGrasp.__new__.__defaults__ = (None,) * len(OIShapeGrasp._fields)


@DATASET.register_module(force=True)
class OIShape(GraspData):

    def __init__(self, cfg):
        super(OIShape, self).__init__(cfg)

    def _preload(self):
        self.grasp_tuple = OIShapeGrasp
        os.environ["OAKINK_DIR"] = os.path.join(self.data_root, "OakInk")
        self.base_dataset = OakInkShape(
            category=self.cfg.OBJ_CATES,
            intent_mode=self.cfg.INTENT_MODE,
            data_split=self.cfg.DATA_SPLIT,
            use_cache=self.use_cache,
            use_downsample_mesh=True,
            preload_obj=False,
        )
        self.action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

    def _init_grasp(self):
        self.grasp_list = self.base_dataset.grasp_list

    def _init_obj_warehouse(self):
        self.obj_warehouse = self.base_dataset.obj_warehouse

    def __len__(self):
        return len(self.base_dataset)

    def get_obj_id(self, idx):
        return self.grasp_list[idx]["obj_id"]

    def get_obj_verts(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).vertices, dtype=np.float32)

    def get_obj_faces(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).faces).astype(np.int32)

    def get_obj_normals(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).vertex_normals, dtype=np.float32)

    def get_joints_obj(self, idx):
        return self.grasp_list[idx]["joints"]

    def get_hand_shape(self, idx):
        return self.grasp_list[idx]["hand_shape"]

    def get_hand_pose_obj(self, idx):
        return self.grasp_list[idx]["hand_pose"]

    def get_obj_rotmat(self, idx):
        return np.eye(3, dtype=np.float32)

    def get_intent(self, idx):
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        return int(act_id), intent_name

    def get_handover(self, idx):
        alt_j, alt_v, alt_pose, alt_shape, alt_tsl = self.base_dataset.get_hand_over(idx)
        return alt_j, alt_v, alt_pose, alt_shape, alt_tsl

    def get_sample_identifier(self, idx):
        cate_id = self.grasp_list[idx]["cate_id"]
        obj_id = self.grasp_list[idx]["obj_id"]
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        subject_id = self.grasp_list[idx]["subject_id"]
        seq_ts = self.grasp_list[idx]["seq_ts"]
        return (f"{self.name}_{self.data_split}_CATE_{cate_id}"
                f"_OBJ({obj_id})_INT({intent_name})_SUB({subject_id})_TS({seq_ts})")

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def main(args):
    OISHAPE_CONFIG = dict(
        DATA_SPLIT=args.data_split,
        DATA_ROOT="data",
        OBJ_CATES=args.categories,
        INTENT_MODE=args.intent_mode,
        DATA_PRESET=dict(
            CENTER_IDX=9,
            USE_CACHE=True,
            N_RESAMPLED_OBJ_POINTS=4096,
        ),
    )
    cfg = CN(OISHAPE_CONFIG)
    dataset: OIShape = OIShape(cfg)
    dataloader = iter(dataset)
    mano_layer = ManoLayer(center_idx=dataset.center_idx, mano_assets_root="assets/mano_v1_2")
    hand_faces = mano_layer.get_mano_closed_faces().numpy()

    viz_context = VizContext()
    viz_context.init(point_size=10.0)

    def next_sample(_):
        grasp_item = next(dataloader)
        hand_pose_obj = grasp_item[Queries.HAND_POSE_OBJ]
        hand_shape = grasp_item[Queries.HAND_SHAPE]
        hand_verts = mano_layer(
            torch.from_numpy(hand_pose_obj).unsqueeze(0),
            torch.from_numpy(hand_shape).unsqueeze(0)).verts.squeeze(0).numpy()

        joint_obj = grasp_item[Queries.JOINTS_OBJ]
        root_joint_obj = joint_obj[dataset.center_idx, :]
        hand_verts_obj = hand_verts + root_joint_obj
        viz_context.update_by_mesh("hand", hand_verts_obj, hand_faces, vcolors=[0.4, 0.8, 0.9], update=True)

        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS]
        obj_normals_ds = grasp_item[Queries.OBJ_NORMALS_OBJ_DS]
        viz_context.update_by_pc("obj", obj_verts_ds, obj_normals_ds, pcolors=[1.0, 0.57, 0.61], update=True)

    next_sample(None)
    viz_context.register_key_callback("D", next_sample)

    viz_context.run()
    viz_context.deinit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="viz grabnet grasp")
    parser.add_argument("--data_dir", type=str, default="data/OakInk", help="environment variable 'OAKINK_DIR'")
    parser.add_argument("--categories", type=str, default="all", help="list of object categories")
    parser.add_argument("--intent_mode",
                        type=list,
                        action="append",
                        default=["use"],
                        choices=["use", "hold", "liftup", "handover"],
                        help="intent mode, list of intents")
    parser.add_argument("--data_split",
                        type=str,
                        default="train",
                        choices=["train", "test", "val", "all"],
                        help="data split")

    args = parser.parse_args()
    main(args)
