from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes

from lib.datasets.grasp_query import Queries
from lib.utils.builder import MODEL
from lib.utils.pcd import point2point_signed
from lib.utils.transform import aa_to_rotmat

from .grabnet_arch import ResBlock, parms_decode
from .refine_net import RefineNet


@MODEL.register_module()
class RefineHandoverNet(RefineNet):

    def __init__(self, cfg, in_size=778 * 2 + 16 * 6 + 3, h_size=512, n_iters=3, strict=True):
        super(RefineHandoverNet, \
              self).__init__(cfg=cfg, in_size=in_size, h_size=h_size, n_iters=n_iters, strict=strict)

    def _build_modules(self):
        h_size = self.h_size
        in_size = self.in_size

        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = ResBlock(in_size, h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        res = {}

        hand_shape = inp[Queries.HAND_SHAPE]  # (B, 10)
        verts_rhand_f = inp[Queries.HAND_VERTS_OBJ + "_f"]  # (B, 778, 3)
        trans_rhand_f = inp[Queries.HAND_TRANSL_OBJ + "_f"]  # (B, 3)
        global_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, :3]  # (B, 3)
        fpose_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, 3:]  # (B, 45)
        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device

        verts_obj = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, 10000, 3)
        verts_rhand = inp[Queries.HAND_VERTS_OBJ]  # (B, 778, 3)
        alt_verts_rhand = inp[Queries.ALT_HAND_VERTS_OBJ]  # (B, 778, 3)
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)

        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)

        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=verts_rhand, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)

        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_obj, rh_mesh)
        o2h_signed_gt, h2o_gt, _ = point2point_signed(verts_rhand, verts_obj, rh_mesh_gt)

        h2h_signed, h2h, _ = point2point_signed(verts_rhand_f, alt_verts_rhand, rh_mesh)
        h2h_signed_gt, h2h_gt, _ = point2point_signed(verts_rhand, alt_verts_rhand, rh_mesh_gt)

        h2o_dist = h2o.abs()
        h2o_dist_gt = h2o_gt.abs()
        h2h_dist = h2h.abs()
        h2h_dist_gt = h2h_gt.abs()

        inp["h2o_dist_gt"] = h2o_dist_gt
        inp["h2h_dist_gt"] = h2h_dist_gt
        inp["o2h_gt"] = o2h_signed_gt
        res["h2o_dist"] = h2o_dist
        res["h2h_dist"] = h2h_dist

        hand_param = self._forward_impl(
            h2o_dist=h2o,
            h2h_dist=h2h,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_obj,
            alt_verts_rhand=alt_verts_rhand,
            hand_shape=hand_shape,
        )
        res.update(hand_param)

        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_out: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        res["hand_verts"] = mano_out.verts + hand_param["transl"].unsqueeze(1)

        rnet_loss = self.compute_loss(res, inp, **kwargs)
        if step_idx % self.log_freq == 0:
            for k, v in rnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, rnet_loss

    def validation_step(self, inp, step_idx, mode="val", **kwargs):
        return self.training_step(inp, step_idx, mode=mode, **kwargs)

    def testing_step(self, inp, cnet_res, step_idx, **kwargs):
        res = {}

        verts_rhand_f = cnet_res[f"Coarse.hand_verts"]  # (B, 778, 3)
        trans_rhand_f = cnet_res[f"Coarse.transl"]  # (B, 3)
        global_hand_pose_obj_f = cnet_res[f"Coarse.global_orient"]  # (B, 3)
        fpose_hand_pose_obj_f = cnet_res[f"Coarse.hand_pose"]  # (B, 45)
        verts_obj = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, NO, 3)
        alt_verts_rhand = inp[Queries.ALT_HAND_VERTS_OBJ]  # (B, 778, 3)
        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)
        hand_shape = torch.zeros(bs, 10).to(device)

        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)

        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_obj, rh_mesh)
        h2h_signed, h2h, _ = point2point_signed(verts_rhand_f, alt_verts_rhand, rh_mesh)

        h2o_dist = h2o.abs()
        h2h_dist = h2h.abs()

        hand_param = self._forward_impl(
            h2o_dist=h2o_dist,
            h2h_dist=h2h_dist,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_obj,
            alt_verts_rhand=alt_verts_rhand,
            hand_shape=hand_shape,
        )

        res.update(hand_param)
        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        verts_rhand = mano_output.verts + hand_param["transl"].unsqueeze(1)
        res["hand_verts"] = verts_rhand

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}

    def _forward_impl(
        self,
        h2o_dist,
        h2h_dist,
        fpose_rhand_rotmat_f,
        trans_rhand_f,
        global_orient_rhand_rotmat_f,
        verts_object,
        alt_verts_rhand,
        hand_shape,
        **kwargs,
    ):
        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f

        for i in range(self.n_iters):

            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                mano_pose = torch.cat([hand_parms["global_orient"], hand_parms["hand_pose"]], dim=1)  # (B, 48)
                mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)

                verts_rhand = mano_output.verts + hand_parms["transl"].unsqueeze(1)
                _, h2o_dist, _ = point2point_signed(verts_rhand, verts_object)
                _, h2h_dist, _ = point2point_signed(verts_rhand, alt_verts_rhand)

            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, h2h_dist, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            pose = self.out_p(X)
            trans = self.out_t(X)

            init_trans = init_trans + trans
            init_pose = init_pose + pose

        hand_parms = parms_decode(init_pose, init_trans)
        return hand_parms

    def compute_loss(self, prd, tgt, **kwargs):
        device = tgt[Queries.HAND_VERTS_OBJ].device
        dtype = tgt[Queries.HAND_VERTS_OBJ].dtype
        bs = tgt[Queries.HAND_VERTS_OBJ].shape[0]
        verts_object = tgt[Queries.OBJ_VERTS_OBJ_DS]  # (B, OV, 3)
        alt_verts_rhand = tgt[Queries.ALT_HAND_VERTS_OBJ]  # (B, 778, 3)

        if self.v_weights.device != device:
            self.v_weights = self.v_weights.to(device)
            self.v_weights2 = self.v_weights2.to(device)
            self.vpe = self.vpe.to(device)

        verts_rhand = prd["hand_verts"]
        rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)

        o2h_signed, h2o, _ = point2point_signed(verts_rhand, verts_object, rh_mesh)
        h2h_signed, h2h, _ = point2point_signed(verts_rhand, alt_verts_rhand, rh_mesh)

        loss_dist_h = (
            35 * (1.0 - self.kl_coef) *
            torch.mean(torch.einsum("ij,j->ij", torch.abs(h2o.abs() - tgt['h2o_dist_gt'].abs()), self.v_weights)))
        loss_dist_h2h = (
            35 * (1.0 - self.kl_coef) *
            torch.mean(torch.einsum("ij,j->ij", torch.abs(h2h.abs() - tgt['h2h_dist_gt'].abs()), self.v_weights)))
        ########## verts loss
        loss_mesh_rec_w = (20 * (1.0 - self.kl_coef) * torch.mean(
            torch.einsum("ijk,j->ijk", torch.abs((tgt[Queries.HAND_VERTS_OBJ] - verts_rhand)), self.v_weights2)))
        ########## edge loss
        loss_edge = 30 * (1. - self.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe),
                                                           self.edges_for(tgt[Queries.HAND_VERTS_OBJ], self.vpe))

        loss_dict = {
            "loss_edge_r": loss_edge,
            "loss_mesh_rec_r": loss_mesh_rec_w,
            "loss_dist_h_r": loss_dist_h,
            "loss_dist_h2h_r": loss_dist_h2h,
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total_r'] = loss_total
        loss_dict['loss'] = loss_total
        return loss_dict
