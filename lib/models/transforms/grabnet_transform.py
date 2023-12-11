import warnings

import numpy as np
import torch
import torch.nn as nn
from bps_torch.bps import bps_torch
from manotorch.manolayer import ManoLayer, MANOOutput
from torch.distributions import Normal

from lib.datasets.grasp_query import Queries
from lib.models.layers.pose_disturber import PoseDisturber
from lib.utils.builder import TRANSFORM
from lib.utils.transform import (aa_to_rotmat, ee_to_rotmat, mano_to_openpose, rotmat_to_aa)


@TRANSFORM.register_module()
class GrabNetTransformObject(nn.Module):

    def __init__(self, cfg):
        super(GrabNetTransformObject, self).__init__()
        self.name = self.__class__.__name__
        self.cfg = cfg
        self.preset_cfg = cfg.DATA_PRESET
        self.rand_rot: bool = cfg.RAND_ROT
        self.rand_rad_std = cfg.get("RAND_DEG_STD", 60) * np.pi / 180  # sigma

        self.use_original_obj_rot: bool = cfg.USE_ORIGINAL_OBJ_ROT
        if self.rand_rot and self.use_original_obj_rot:
            warnings.warn("RAND_ROT and USE_ORIGINAL_OBJ_ROT are both True, use ORIGINAL_OBJ_ROT only")
            self.rand_rot = False

        if self.rand_rot is True:
            self.random_rot_distb = Normal(torch.tensor(0.0), torch.tensor(self.rand_rad_std))
        self.bps_basis = torch.from_numpy(np.load(cfg.BPS_BASIS_PATH)["basis"]).to(torch.float32)
        self.bps_feat_type = cfg.BPS_FEAT_TYPE
        self.bps_layer = bps_torch(custom_basis=self.bps_basis)
        self.dummy_param = nn.Parameter(torch.empty(0))

    @torch.no_grad()
    def forward(self, batch):
        # device = self.dummy_param.device
        # batch = batch_to_device(batch, device)
        batch_size = batch[Queries.OBJ_VERTS_OBJ].shape[0]
        device = batch[Queries.OBJ_VERTS_OBJ].device

        # disturb obj's rotation in obj space
        if self.use_original_obj_rot is True:
            # @NOTE use rotation from dataset, typical at testing step.
            obj_rotmat = batch[Queries.OBJ_ROTMAT]
        elif self.rand_rot is True:
            # @NOTE set obj_rotmat as random rot, typical at training step.
            rand_rot_euler = self.random_rot_distb.sample((batch_size, 3)).float().to(device)
            bound = min(np.pi, 3 * self.rand_rad_std)
            rand_rot_euler = rand_rot_euler.clamp(-bound, bound)
            rand_rotmat = ee_to_rotmat(rand_rot_euler, convention='xyz')
            obj_rotmat = rand_rotmat
        else:
            # @NOTE set obj_rotmat as identity
            obj_rotmat = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        batch[Queries.OBJ_ROTMAT] = obj_rotmat

        # obj original verts, without downsampled
        obj_verts_obj = batch[Queries.OBJ_VERTS_OBJ]
        obj_verts_obj_new = torch.matmul(obj_rotmat, obj_verts_obj.transpose(1, 2)).transpose(1, 2)
        batch[Queries.OBJ_VERTS_OBJ] = obj_verts_obj_new  # @OVERWRITE to batch

        obj_verts_ds = batch[Queries.OBJ_VERTS_OBJ_DS]
        obj_normals_ds = batch[Queries.OBJ_NORMALS_OBJ_DS]
        obj_verts_ds_new = torch.matmul(obj_rotmat, obj_verts_ds.transpose(1, 2)).transpose(1, 2)
        obj_normals_ds_new = torch.matmul(obj_rotmat, obj_normals_ds.transpose(1, 2)).transpose(1, 2)
        batch[Queries.OBJ_VERTS_OBJ_DS] = obj_verts_ds_new  # @OVERWRITE to batch
        batch[Queries.OBJ_NORMALS_OBJ_DS] = obj_normals_ds_new  # @OVERWRITE to batch

        # change verts into bps
        obj_bps = self.bps_layer.encode(obj_verts_ds_new, feature_type=self.bps_feat_type)[self.bps_feat_type]
        batch[Queries.OBJ_BPS] = obj_bps.to(device)

        return batch


@TRANSFORM.register_module()
class GrabNetTransformHandObject(GrabNetTransformObject):

    def __init__(self, cfg):
        super(GrabNetTransformHandObject, self).__init__(cfg)
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
        if cfg.get("POSE_DISTURBER") is not None:
            self.pose_disturber = PoseDisturber(**cfg.POSE_DISTURBER)
        else:
            self.pose_disturber = None

    @torch.no_grad()
    def forward(self, batch):
        batch = super(GrabNetTransformHandObject, self).forward(batch)
        obj_rotmat = batch[Queries.OBJ_ROTMAT]  # get obj_rotmat from super.forward()
        batch_size = obj_rotmat.shape[0]
        device = obj_rotmat.device

        hand_pose_obj = batch[Queries.HAND_POSE_OBJ]  # (B, 48), only support axisang form
        hp_glob = hand_pose_obj[:, :3]  # (B, 3)
        hp_glob_new = rotmat_to_aa(torch.matmul(obj_rotmat, aa_to_rotmat(hp_glob)))
        hand_pose_obj_new = torch.cat([hp_glob_new, hand_pose_obj[:, 3:]], dim=1)  # (B, 48)
        batch[Queries.HAND_POSE_OBJ] = hand_pose_obj_new  # @OVERWRITE

        joints_obj = batch[Queries.JOINTS_OBJ]  # (B, 21, 3)
        joints_obj_new = torch.matmul(obj_rotmat, joints_obj.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 3)
        batch[Queries.JOINTS_OBJ] = joints_obj_new  # @OVERWRITE

        hand_shape = batch[Queries.HAND_SHAPE]  # (B, 10)
        mano_output: MANOOutput = self.mano_layer(hand_pose_obj_new, hand_shape)

        if self.center_idx is not None:
            hand_transl_obj_new = joints_obj_new[:, self.center_idx, :]
        else:
            hand_transl_obj_new = joints_obj_new - mano_to_openpose(self.mano_layer.th_J_regressor, mano_output.verts)
            hand_transl_obj_new = torch.mean(hand_transl_obj_new, dim=1, keepdim=False)  # (B, 3)

        batch[Queries.HAND_TRANSL_OBJ] = hand_transl_obj_new  # @OVERWRITE
        hand_verts_obj = mano_output.verts + hand_transl_obj_new.unsqueeze(1)  # (B, 778, 3)
        batch[Queries.HAND_VERTS_OBJ] = hand_verts_obj

        # region ===== disturb hand pose and generate hand joints and verts >>>>>
        hand_pose_key = f"{Queries.HAND_POSE_OBJ}_f"
        hand_joint_key = f"{Queries.JOINTS_OBJ}_f"
        hand_transl_key = f"{Queries.HAND_TRANSL_OBJ}_f"
        hand_verts_key = f"{Queries.HAND_VERTS_OBJ}_f"

        if hand_pose_key in batch:  # @NOTE Use the perturbed pose & joint from outside
            hand_pose_f = batch[hand_pose_key]
            hand_joint_f = batch[hand_joint_key]

            # apply obj_rot to the "_f" pose and joints
            hand_pose_glob_f = rotmat_to_aa(torch.matmul(obj_rotmat, aa_to_rotmat(hand_pose_f[:, :3])))
            hand_pose_f = torch.cat([hand_pose_glob_f, hand_pose_f[:, 3:]], dim=1)  # (B, 48)
            hand_joint_f = torch.matmul(obj_rotmat, hand_joint_f.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 3)

            batch[hand_pose_key] = hand_pose_f  # @OVERWRITE
            mano_output_f: MANOOutput = self.mano_layer(hand_pose_f, hand_shape)

            if self.center_idx is not None:
                hand_transl_f = hand_joint_f[:, self.center_idx, :]
            else:
                hand_transl_f = hand_joint_f - mano_to_openpose(self.mano_layer.th_J_regressor, mano_output.verts)
                hand_transl_f = torch.mean(hand_transl_f, dim=1, keepdim=False)  # (B, 3)

            batch[hand_transl_key] = hand_transl_f

            # use the perturbed hand pose to generate the perturbed hand joints and verts
            batch[hand_joint_key] = mano_output_f.joints + hand_transl_f.unsqueeze(1)
            batch[hand_verts_key] = mano_output_f.verts + hand_transl_f.unsqueeze(1)
        elif self.pose_disturber is not None:  # @NOTE Do perturbation online
            hand_pose_f, hand_transl_f = self.pose_disturber(hand_pose_obj_new.clone(), hand_transl_obj_new.clone())

            batch[hand_pose_key] = hand_pose_f
            batch[hand_transl_key] = hand_transl_f

            mano_output_f: MANOOutput = self.mano_layer(hand_pose_f, hand_shape)

            batch[hand_joint_key] = mano_output_f.joints + hand_transl_f.unsqueeze(1)
            batch[hand_verts_key] = mano_output_f.verts + hand_transl_f.unsqueeze(1)
        # endregion <<<<<

        hand_faces = self.mano_layer.th_faces
        batch[Queries.HAND_FACES] = hand_faces.repeat(batch_size, 1, 1).to(device)
        return batch


@TRANSFORM.register_module()
class GrabNetTransformIntent(GrabNetTransformHandObject):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_intents = cfg.DATA_PRESET.N_INTENTS

    @torch.no_grad()
    def forward(self, batch):
        batch = super(GrabNetTransformIntent, self).forward(batch)
        intent_id = batch[Queries.INTENT_ID]  # (B, )
        batch_size = intent_id.shape[0]
        # intent_vec = torch.zeros((batch_size, self.intentD), dtype=torch.float32)
        # intent_vec[torch.arange(batch_size), intent_id] = 1.0
        # batch[Queries.INTENT_VEC] = intent_vec.to(intent_id.device)

        intent_vec = torch.zeros((batch_size, self.n_intents), dtype=torch.float32)
        # convert intet_id to onehot
        intent_vec[torch.arange(batch_size), intent_id - 1] = 1.0  # intent id starts from 1
        batch[Queries.INTENT_VEC] = intent_vec.to(intent_id.device)
        return batch


@TRANSFORM.register_module()
class GrabNetTransformHandover(GrabNetTransformHandObject):

    @torch.no_grad()
    def forward(self, batch):
        batch = super(GrabNetTransformHandover, self).forward(batch)
        obj_rotmat = batch[Queries.OBJ_ROTMAT]  # get obj_rotmat from super.forward()

        alt_hand_pose = batch[Queries.ALT_HAND_POSE_OBJ]  # (B, 48), only support axisang form
        alt_hp_glob = alt_hand_pose[:, :3]  # (B, 3)
        alt_hp_glob_new = rotmat_to_aa(torch.matmul(obj_rotmat, aa_to_rotmat(alt_hp_glob)))
        alt_hand_pose_new = torch.cat([alt_hp_glob_new, alt_hand_pose[:, 3:]], dim=1)  # (B, 48)
        batch[Queries.ALT_HAND_POSE_OBJ] = alt_hand_pose_new  # @OVERWRITE

        alt_joint = batch[Queries.ALT_JOINTS_OBJ]  # (B, 21, 3)
        alt_joint_new = torch.matmul(obj_rotmat, alt_joint.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 3)
        batch[Queries.ALT_JOINTS_OBJ] = alt_joint_new  # @OVERWRITE

        alt_verts = batch[Queries.ALT_HAND_VERTS_OBJ]
        alt_verts_new = torch.matmul(obj_rotmat, alt_verts.permute(0, 2, 1)).permute(0, 2, 1)
        batch[Queries.ALT_HAND_VERTS_OBJ] = alt_verts_new  # @OVERWRITE

        # @NOTE: carefully deal with hand translation
        if self.center_idx is not None:
            alt_hand_transl = alt_joint_new[:, self.center_idx, :]
        else:
            alt_hand_transl = (self.mano_layer(alt_hand_pose_new, batch[Queries.ALT_HAND_SHAPE]).joints -
                               mano_to_openpose(self.mano_layer.th_J_regressor, alt_verts_new))
            alt_hand_transl = torch.mean(alt_hand_transl, dim=1, keepdim=False)
        batch[Queries.ALT_HAND_TRANSL_OBJ] = alt_hand_transl
        return batch
