from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes

from lib.datasets.grasp_query import Queries
from lib.metrics.basic_metric import LossMetric
from lib.models.model_abc import ModelABC
from lib.utils.builder import MODEL
from lib.utils.net_utils import load_weights
from lib.utils.pcd import point2point_signed
from lib.utils.transform import aa_to_rotmat

from .grabnet_arch import ResBlock, parms_decode


@MODEL.register_module()
class RefineNet(ModelABC):

    def __init__(self, cfg, in_size=778 + 16 * 6 + 3, h_size=512, n_iters=3, strict=False):
        super(RefineNet, self).__init__()
        self.stage = "Refine"
        self.cfg = cfg
        self.n_iters = n_iters
        self.h_size = h_size
        self.in_size = in_size

        self.vpe = torch.from_numpy(np.load(cfg.VPE_PATH)).to(torch.long)
        self.v_weights = torch.from_numpy(np.load(cfg.C_WEIGHT_PATH)).to(torch.float32)
        self.v_weights2 = torch.pow(self.v_weights, 1.0 / 2.5)
        self.contact_v = self.v_weights > 0.8
        self.kl_coef = cfg.KL_COEF

        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

        self._build_layers()
        self._build_loss()
        self._build_evaluation()
        load_weights(self, pretrained=cfg.PRETRAINED, strict=strict)

    def _build_layers(self):
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

    def _build_loss(self):
        self.has_loss = True
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.loss_metric = LossMetric(self.cfg)

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq

    def forward(self, inp, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inp, step_idx=step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inp, step_idx=step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inp, step_idx=step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        res = {}

        # train RefineNet with _f: fake inp data
        verts_rhand_f = inp[Queries.HAND_VERTS_OBJ + "_f"]  # (B, 778, 3)
        trans_rhand_f = inp[Queries.HAND_TRANSL_OBJ + "_f"]  # (B, 3)
        global_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, :3]  # (B, 3)
        fpose_hand_pose_obj_f = inp[Queries.HAND_POSE_OBJ + "_f"][:, 3:]  # (B, 45)

        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device
        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)
        verts_object = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, 10000, 3)
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_object, rh_mesh)
        h2o_dist = h2o.abs()
        res["h2o_dist"] = h2o_dist
        hand_shape = inp[Queries.HAND_SHAPE]
        verts_rhand = inp[Queries.HAND_VERTS_OBJ]  # (B, 778, 3)
        rh_mesh_gt = Meshes(verts=verts_rhand, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed_gt, h2o_gt, _ = point2point_signed(verts_rhand, verts_object, rh_mesh_gt)
        h2o_dist_gt = h2o_gt.abs()

        inp["h2o_dist_gt"] = h2o_dist_gt  # for compute loss

        hand_param = self._forward_impl(
            h2o_dist=h2o_dist,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_object,
            hand_shape=hand_shape,
        )

        res.update(hand_param)
        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        verts_rhand = mano_output.verts + hand_param["transl"].unsqueeze(1)
        res["hand_verts"] = verts_rhand  # for visualize

        rnet_loss = self.compute_loss(res, inp, **kwargs)
        self.loss_metric.feed(rnet_loss)
        if step_idx % self.log_freq == 0:
            for k, v in rnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, rnet_loss

    def on_train_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"  # GrabNet-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validation_step(self, inp, step_idx, mode="val", **kwargs):
        return self.training_step(inp, step_idx, mode=mode, **kwargs)

    def on_val_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"  # GrabNet-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def testing_step(self, inp, cnet_res, step_idx, **kwargs):
        res = {}

        # for k, v in cnet_res.items():
        verts_rhand_f = cnet_res[f"Coarse.hand_verts"]  # (B, 778, 3)
        trans_rhand_f = cnet_res[f"Coarse.transl"]  # (B, 3)
        global_hand_pose_obj_f = cnet_res[f"Coarse.global_orient"]  # (B, 3)
        fpose_hand_pose_obj_f = cnet_res[f"Coarse.hand_pose"]  # (B, 45)

        bs = verts_rhand_f.shape[0]
        device = verts_rhand_f.device
        global_orient_rhand_rotmat_f = aa_to_rotmat(global_hand_pose_obj_f).unsqueeze(1)  # (B, 1, 3, 3)
        fpose_rhand_rotmat_f = aa_to_rotmat(fpose_hand_pose_obj_f.reshape(-1, 3)).reshape(bs, -1, 3, 3)  # (B, 15, 3, 3)
        verts_object = inp[Queries.OBJ_VERTS_OBJ_DS]  # (B, 10000, 3)
        rhfaces = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand_f, faces=rhfaces).to(device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand_f, verts_object, rh_mesh)
        h2o_dist = h2o.abs()
        res["h2o_dist"] = h2o_dist
        hand_shape = torch.zeros(bs, 10).to(device)

        hand_param = self._forward_impl(
            h2o_dist=h2o_dist,
            fpose_rhand_rotmat_f=fpose_rhand_rotmat_f,
            trans_rhand_f=trans_rhand_f,
            global_orient_rhand_rotmat_f=global_orient_rhand_rotmat_f,
            verts_object=verts_object,
            hand_shape=hand_shape,
        )

        res.update(hand_param)
        mano_pose = torch.cat([hand_param["global_orient"], hand_param["hand_pose"]], dim=1)  # (B, 48)
        mano_output: MANOOutput = self.mano_layer(mano_pose, hand_shape)
        verts_rhand = mano_output.verts + hand_param["transl"].unsqueeze(1)
        res["hand_verts"] = verts_rhand  # for visualize

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}

    def _forward_impl(self, h2o_dist, fpose_rhand_rotmat_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object,
                      **kwargs):

        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f

        for i in range(self.n_iters):
            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                mano_pose = torch.cat([hand_parms["global_orient"], hand_parms["hand_pose"]], dim=1)  # (B, 48)
                mano_shape = kwargs[Queries.HAND_SHAPE]  # (B, 10)
                mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)

                verts_rhand = mano_output.verts + hand_parms["transl"].unsqueeze(1)
                _, h2o_dist, _ = point2point_signed(verts_rhand, verts_object)

            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_pose, init_trans], dim=1)
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

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def compute_loss(self, prd, tgt, **kwargs):
        device = tgt[Queries.HAND_VERTS_OBJ].device
        dtype = tgt[Queries.HAND_VERTS_OBJ].dtype
        bs = tgt[Queries.HAND_VERTS_OBJ].shape[0]

        if self.v_weights.device != device:
            self.v_weights = self.v_weights.to(device)
            self.v_weights2 = self.v_weights2.to(device)
            self.vpe = self.vpe.to(device)

        verts_rhand = prd["hand_verts"]
        rh_f = self.mano_layer.th_faces.expand(bs, -1, -1)
        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]
        o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
        ######### dist loss
        loss_dist_h = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o.abs() - tgt['h2o_dist_gt'].abs()), self.v_weights2))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ijk,j->ijk', torch.abs(verts_rhand - tgt[Queries.HAND_VERTS_OBJ]), self.v_weights2))
        ########## edge loss
        loss_edge = 30 * (1. - self.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe),
                                                           self.edges_for(tgt[Queries.HAND_VERTS_OBJ], self.vpe))
        ##########

        loss_dict = {
            'loss_edge_r': loss_edge,
            'loss_mesh_rec_r': loss_mesh_rec_w,
            'loss_dist_h_r': loss_dist_h,
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total_r'] = loss_total
        loss_dict['loss'] = loss_total
        return loss_dict
