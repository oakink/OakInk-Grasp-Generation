from abc import ABCMeta
from collections import namedtuple
from typing import Dict, List

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from torch.utils.data._utils.collate import default_collate

from lib.datasets.grasp_query import Queries, match_collate_queries
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger


class GraspData(metaclass=ABCMeta):

    def __init__(self, cfg):
        super().__init__()
        self.name = self.__class__.__name__
        self.cfg = cfg
        self.version = cfg.get("VERSION", "")
        self.data_root = cfg.DATA_ROOT
        self.data_split = cfg.DATA_SPLIT
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.data_mode = cfg.get("DATA_MODE", "grasp")
        self.pre_load_obj = cfg.get("PRE_LOAD_OBJ", False)
        self.use_cache = cfg.DATA_PRESET.get("USE_CACHE", False)
        self.filter_no_contact = cfg.DATA_PRESET.get("FILTER_NO_CONTACT", False)
        self.filter_no_contact_thresh = cfg.DATA_PRESET.get("FILTER_NO_CONTACT_THRESH", 5.0)
        self.n_points = cfg.DATA_PRESET.get("N_RESAMPLED_OBJ_POINTS", 2048)
        self.side = "right"
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side=self.side,
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

        self._preload()
        self._init_obj_warehouse()
        self._init_grasp()

    def _preload(self):
        self.grasp_tuple = namedtuple("MetaGrasp", None)

    def _init_grasp(self):
        self.grasp_list = list()

    def _init_obj_warehouse(self):
        self.obj_warehouse = dict()

    def _logging_info(self):
        logger.info(f"{self.name}-{self.data_split} has {len(self.grasp_list)} samples")

    def __len__(self):
        return len(self.grasp_list)

    def _filter_no_contact(self, grasp_list: List):
        nfiltered = 0

        filtered_grasp_list = []
        for _, g in enumerate(etqdm(grasp_list, desc=f"Filter {self.name} grasps")):
            grasp = self.grasp_tuple(**g)

            # filter no contact
            if self.filter_no_contact:
                # naive
                obj_verts_obj = self.obj_warehouse[grasp.obj_id].vertices  # (NVERTS, 3)
                joints_obj = grasp.joints_obj  # (21, 3)
                min_dist = np.min(cdist(obj_verts_obj, joints_obj) * 1000.0)
                if min_dist > self.filter_no_contact_thresh:
                    nfiltered += 1
                    continue

            filtered_grasp_list.append(grasp)
        return filtered_grasp_list

    def get_sample_identifier(self, idx):
        raise NotImplementedError

    def get_obj_id(self, idx):
        return self.grasp_list[idx].obj_id

    def get_joints_obj(self, idx):
        return self.grasp_list[idx].joints_obj

    def get_hand_shape(self, idx):
        return self.grasp_list[idx].hand_shape

    def get_hand_pose_obj(self, idx):
        return self.grasp_list[idx].hand_pose_obj

    def get_intent(self, idx):
        raise NotImplementedError(f"{self.name} not support intent")

    def get_handover(self, idx):
        raise NotImplementedError(f"{self.name} not support handover")

    def get_obj_rotmat(self, idx):
        return self.grasp_list[idx].obj_rot

    def get_obj_verts(self, idx):
        raise NotImplementedError()

    def get_obj_faces(self, idx):
        raise NotImplementedError()

    def get_obj_normals(self, idx):
        raise NotImplementedError()

    def process_obj_pack(self, obj_verts, obj_faces, n_sample_verts):
        mesh = Meshes(verts=torch.from_numpy(obj_verts).unsqueeze(0), faces=torch.from_numpy(obj_faces).unsqueeze(0))
        obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, n_sample_verts, return_normals=True)
        obj_pack = dict(
            obj_verts=obj_verts,
            obj_faces=obj_faces,
            obj_verts_ds=obj_verts_ds.squeeze(0).numpy(),
            obj_normals_ds=obj_normals_ds.squeeze(0).numpy(),
            # ...
        )
        return obj_pack

    def process_hand_pack(self, pose, shape, joints_obj):
        joints_in_obj_sys = joints_obj
        mano_out = self.mano_layer(torch.from_numpy(pose[None, ...]), torch.from_numpy(shape[None, ...]))
        joints_in_mano_sys = mano_out.joints.squeeze(0).numpy()  # (21, 3)
        transl = np.mean(joints_in_obj_sys - joints_in_mano_sys, axis=0, keepdims=True)
        verts_in_obj_sys = mano_out.verts.squeeze(0) + transl  # (778, 3)

        hand_pack = dict(
            hand_joints=joints_in_obj_sys,
            hand_verts=verts_in_obj_sys,
            hand_transl=transl,
            hand_pose=pose,
            hand_shape=shape,
            # ...
        )
        return hand_pack

    def __getitem__(self, idx):
        if self.data_mode == "obj":
            return self.getitem_obj(idx)
        elif self.data_mode == "grasp":
            return self.getitem_grasp(idx)
        elif self.data_mode == "intent":
            return self.getitem_intent(idx)
        elif self.data_mode == "handover":
            return self.getitem_handover(idx)
        else:
            raise ValueError(f"Unknown data mode {self.data_mode}")

    def getitem_obj(self, idx):
        sample = {}

        sample[Queries.SAMPLE_IDENTIFIER] = self.get_sample_identifier(idx)
        sample[Queries.OBJ_ID] = self.get_obj_id(idx)

        # original, but may not valid
        obj_verts_obj = self.get_obj_verts(idx)
        obj_faces = self.get_obj_faces(idx)

        proc_obj = self.process_obj_pack(obj_verts_obj, obj_faces, self.n_points)
        obj_verts_obj_ds = proc_obj["obj_verts_ds"]  # re-sampled
        obj_normals_obj_ds = proc_obj["obj_normals_ds"]  # re-sampled
        obj_rotmat = self.get_obj_rotmat(idx)

        sample.update({
            Queries.OBJ_VERTS_OBJ: obj_verts_obj,
            Queries.OBJ_VERTS_OBJ_DS: obj_verts_obj_ds,
            Queries.OBJ_NORMALS_OBJ_DS: obj_normals_obj_ds,
            Queries.OBJ_FACES: obj_faces,
            Queries.OBJ_ROTMAT: obj_rotmat,
        })
        return sample

    def getitem_grasp(self, idx):
        sample = self.getitem_obj(idx)
        joints_obj = self.get_joints_obj(idx)
        hand_pose_obj = self.get_hand_pose_obj(idx)
        hand_shape = self.get_hand_shape(idx)
        # @NOTE: we move this to the Transform layer for batch processing !
        # proc_hand = self.process_hand_pack(hand_pose_obj, hand_shape, joints_obj)

        sample.update({
            Queries.JOINTS_OBJ: joints_obj,
            Queries.HAND_POSE_OBJ: hand_pose_obj,
            Queries.HAND_SHAPE: hand_shape,
        })
        return sample

    def getitem_intent(self, idx):
        sample = self.getitem_grasp(idx)
        intent_id, intent_name = self.get_intent(idx)
        sample.update({
            Queries.INTENT_ID: intent_id,
            Queries.INTENT_NAME: intent_name,
        })
        return sample

    def getitem_handover(self, idx):
        sample = self.getitem_grasp(idx)
        alt_j, alt_v, alt_pose, alt_shape, _ = self.get_handover(idx)
        sample.update({
            Queries.ALT_JOINTS_OBJ: alt_j,
            Queries.ALT_HAND_POSE_OBJ: alt_pose,
            Queries.ALT_HAND_SHAPE: alt_shape,
            Queries.ALT_HAND_VERTS_OBJ: alt_v,
        })
        return sample


def grasp_data_collate(batch: List[Dict]):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """
    # *  NEW QUERY: CollateQueries.PADDING_MASK

    extend_queries = {Queries.OBJ_VERTS_OBJ, Queries.OBJ_NORMALS_OBJ, Queries.OBJ_FACES}
    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        padding_query_field = match_collate_queries(pop_query)
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            orig_len = pop_value.shape[0]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
            if padding_query_field not in sample:
                # generate a new field, contains padding mask
                # note that only the beginning pop_value.shape[0] points are in effect
                # so the mask will be a vector of length max_size, with origin_len ones in the beginning
                padding_mask = np.zeros(max_size, dtype=np.int32)
                padding_mask[:orig_len] = 1
                sample[padding_query_field] = padding_mask

    # store the mask filtering the points
    batch = default_collate(batch)  # this function np -> torch
    return batch
