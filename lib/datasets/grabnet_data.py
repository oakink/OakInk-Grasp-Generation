import os
import pickle
from collections import namedtuple
from typing import List

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from trimesh import Trimesh

from lib.datasets.grasp_data import GraspData
from lib.datasets.grasp_query import Queries
from lib.utils.builder import DATASET
from lib.utils.logger import logger

GrabNetGrasp = namedtuple(
    "GrabNetGrasp",
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
GrabNetGrasp.__new__.__defaults__ = (None,) * len(GrabNetGrasp._fields)


@DATASET.register_module(force=True)
class GrabNetData(GraspData):

    def _preload(self):
        self.grasp_tuple = GrabNetGrasp
        self.data_root_dir = os.path.join(self.data_root, f"GrabNet{self.version}_grasp")
        self.obj_models_dir = os.path.join(self.data_root_dir, "object_models")

    def _init_grasp(self):
        annot_split_dir = os.path.join(self.data_root_dir, self.data_split)
        annot_fragment_files = []
        for (dirpath, dirname, filenames) in os.walk(annot_split_dir):
            annot_fragment_files.extend(filenames)
        grasp_list = []
        for i, frag_file in enumerate(annot_fragment_files):
            with open(os.path.join(annot_split_dir, frag_file), "rb") as f:
                grasp_list.extend(pickle.load(f))
        self.grasp_list = self._filter_no_contact(grasp_list)

    def _init_obj_warehouse(self):
        self.obj_warehouse = {}
        if self.pre_load_obj:
            all_obj_models_list = os.listdir(self.obj_models_dir)
            for el in all_obj_models_list:
                obj_id, suffix = el.split(".")[0], el.split(".")[1]
                assert suffix == "obj" or suffix == "ply", f"suffix {suffix} is not supported, use .obj and .ply"
                obj_path = os.path.join(self.obj_models_dir, el)
                obj_mesh = trimesh.load(obj_path, process=False)
                self.obj_warehouse[obj_id] = obj_mesh

    def _filter_no_contact(self, grasp_list: List):
        """GrabNet dataset does not need filtering"""
        res = [self.grasp_tuple(**g) for g in grasp_list]
        return res

    def get_sample_identifier(self, idx):
        split = self.grasp_list[idx].split
        obj_id = self.grasp_list[idx].obj_id
        sbj_id = self.grasp_list[idx].sbj_name
        sample_idx = self.grasp_list[idx].sample_idx
        return f"{self.name}-{split}-obj-{obj_id}-sbj-{sbj_id}-idx-{sample_idx}"

    def _load_obj_mesh(self, idx):
        grasp_item = self.grasp_list[idx]
        obj_id = grasp_item.obj_id
        if obj_id not in self.obj_warehouse:
            obj_path = os.path.join(self.obj_models_dir, f"{obj_id}.ply")
            obj_mesh: Trimesh = trimesh.load(obj_path, process=False)
            self.obj_warehouse[obj_id] = obj_mesh
        else:
            obj_mesh: Trimesh = self.obj_warehouse[obj_id]
        return obj_mesh

    def get_obj_verts(self, idx):
        obj_trimesh: Trimesh = self._load_obj_mesh(idx)
        verts = np.asarray(obj_trimesh.vertices, dtype=np.float32)
        return verts

    def get_obj_faces(self, idx):
        obj_trimesh: Trimesh = self._load_obj_mesh(idx)
        faces = np.asarray(obj_trimesh.faces, dtype=np.int32)
        return faces

    def get_obj_normals(self, idx):
        obj_trimesh: Trimesh = self._load_obj_mesh(idx)
        normals = np.asarray(obj_trimesh.vertex_normals, dtype=np.float32)
        return normals

    def get_obj_rotmat(self, idx):
        return self.grasp_list[idx].obj_rot

    def __getitem__(self, idx):
        return super().__getitem__(idx)


# This class is directly copied from the GrabNet official release.
# DO NOT modify it, inheritance instead.
class _GrabNetOfficial(data.Dataset):
    GRABNET_CENTER_IDX = None

    def __init__(self, dataset_dir, ds_name="train", dtype=torch.float32, only_params=False, load_on_ram=False):

        super().__init__()

        self.only_params = only_params

        # GrabNet/data/train
        self.ds_path = os.path.join(dataset_dir, ds_name)
        # GrabNet/data/train/grabnet_train.npz
        # 这里面是每一帧手的global_rotmat, fpose_rotmat, trans; 物体的trans, rotmat；手_f的global_rotmat,
        # fpose_rotmat和trans
        self.ds = self._np2torch(os.path.join(self.ds_path, "grabnet_%s.npz" % ds_name))

        # GrabNet/data/train/frame_names.npz 每一帧的文件路径
        frame_names = np.load(os.path.join(dataset_dir, ds_name, "frame_names.npz"))["frame_names"]
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        # 每一帧手的名字
        self.frame_sbjs = np.asarray([name.split("/")[-3] for name in self.frame_names])
        # 每一帧物体的名字
        self.frame_objs = np.asarray([name.split("/")[-2].split("_")[0] for name in self.frame_names])

        # 各类手的编号
        self.sbjs = np.unique(self.frame_sbjs)
        # 每种物体的sample_idx和mesh_file，共50种
        self.obj_info = np.load(os.path.join(dataset_dir, "obj_info.npy"), allow_pickle=True).item()
        # 每种手的顶点坐标和mano_beta，共10种
        self.sbj_info = np.load(os.path.join(dataset_dir, "sbj_info.npy"), allow_pickle=True).item()

        # bps_torch data
        bps_fname = os.path.join(dataset_dir, "bps.npz")
        self.bps = torch.from_numpy(np.load(bps_fname)["basis"]).to(dtype)
        # Hand vtemps and betas

        # order: [s1 s10 s2 s3 s4 s5 s6 s7 s8 s9]
        # 每种手的顶点
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]["rh_vtemp"] for sbj in self.sbjs]))  #手的顶点
        # 每种手的mano_beta
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]["rh_betas"] for sbj in self.sbjs]))  #mano beta

        # 将每一帧手的名字换成索引号
        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx
        self.frame_sbjs = torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    # 根据每一帧的路径，把这一帧的物体、手、手_f和bps读进来
    def _np2torch(self, ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k: torch.tensor(data[k]) for k in data.files}
        return data_torch

    # 如果idx是整数，直接读取该帧，如果是数组，就把其中所有帧读取出来
    def load_disk(self, idx):
        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []

        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        return data_out


class _GrabNetReload(_GrabNetOfficial):

    def __init__(self, grabnet_dir, grab_dir, **kwargs):

        grabnet_data_dir = os.path.join(grabnet_dir, "data")
        super().__init__(grabnet_data_dir, **kwargs)

        self.obj_names = np.unique(self.frame_objs)
        mesh_dir = os.path.join(grab_dir, "tools/object_meshes/contact_meshes")
        self.obj_name2_id_mapping = {}
        self.obj_name2_mesh_mapping = {}
        self.sbj_id2_name_mapping = {}

        for obj in self.obj_names:
            obj_frames_id = np.where(self.frame_objs == obj)[0]
            self.obj_name2_id_mapping[obj] = obj_frames_id
            obj_mesh_raw = trimesh.load(os.path.join(mesh_dir, f"{obj}.ply"), process=False)
            self.obj_name2_mesh_mapping[obj] = obj_mesh_raw

        self.obj_name2_sampleIdx_mapping = {}
        for obj_name, info in self.obj_info.items():
            sample_idx = info["verts_sample_id"]
            self.obj_name2_sampleIdx_mapping[obj_name] = list(sample_idx)

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=self.GRABNET_CENTER_IDX,
            flat_hand_mean=True,
        )
