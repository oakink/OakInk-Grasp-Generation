from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes
from torch.nn import functional as F

from lib.datasets.grasp_query import Queries
from lib.metrics.basic_metric import LossMetric
from lib.models.model_abc import ModelABC
from lib.utils.builder import MODEL
from lib.utils.net_utils import load_weights
from lib.utils.pcd import point2point_signed
from lib.utils.transform import aa_to_rotmat

from .grabnet_arch import ResBlock, parms_decode


@MODEL.register_module()
class CoarseNet(ModelABC):

    def __init__(self, cfg, n_neurons=512, latentD=16, in_bps=4096, in_pose=12, strict=False):
        super(CoarseNet, self).__init__()
        self.stage = "Coarse"
        self.cfg = cfg
        self.latentD = latentD
        self.n_neurons = n_neurons
        self.in_bps = in_bps
        self.in_pose = in_pose
        self.vpe = torch.from_numpy(np.load(cfg.VPE_PATH)).to(torch.long)
        self.kl_coef = cfg.KL_COEF
        self.v_weights = torch.from_numpy(np.load(cfg.C_WEIGHT_PATH)).to(torch.float32)
        self.v_weights2 = torch.pow(self.v_weights, 1.0 / 2.5)
        self.contact_v = self.v_weights > 0.8
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX

        self._build_layers()
        self._build_loss()
        self._build_evaluation()
        load_weights(self, cfg.PRETRAINED, strict=strict)

        self.mano_layer = ManoLayer(rot_mode="axisang",
                                    center_idx=self.center_idx,
                                    mano_assets_root="assets/mano_v1_2",
                                    use_pca=False,
                                    flat_hand_mean=True)

    def _build_layers(self):
        """ create all trainable modules  here
        """
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def _build_loss(self):
        self.has_loss = True
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.loss_metric = LossMetric(self.cfg)

    def forward(self, inp, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inp, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inp, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inp, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat):

        bs = bps_object.shape[0]
        X = torch.cat([bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object):

        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        results = parms_decode(pose, trans)
        results["z"] = Zin

        return results

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        bs = bps_object.shape[0]
        trans_rhand = inp[Queries.HAND_TRANSL_OBJ]  # (B, 3)
        global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]  # (B, 3)
        global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)

        res = self._forward_impl(bps_object, trans_rhand, global_orient_rhand_rotmat)
        cnet_loss = self.compute_loss(prd=res, tgt=inp, **kwargs)
        self.loss_metric.feed(cnet_loss)
        if step_idx % self.log_freq == 0:
            for k, v in cnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)
        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, cnet_loss

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

    def testing_step(self, inp, step_idx, **kwargs):
        res = self.sample_poses(inp[Queries.OBJ_BPS], seed=kwargs.get("seed", None))
        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out: MANOOutput = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res[f"transl"].unsqueeze(1)  # (B, 778, 3)
        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}

    def _forward_impl(self, bps_object, trans_rhand, global_orient_rhand_rotmat, **kwargs):
        """
        :param bps_object: bps_delta of object: Nxn_bpsx3
        :param delta_hand_mano: bps_delta of subject, e.g. hand: Nxn_bpsx3
        :param output_type: bps_delta of something, e.g. hand: Nxn_bpsx3
        :return:
        """
        z = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat)
        z_s = z.rsample()
        hand_parms = self.decode(z_s, bps_object)
        results = {"mean": z.mean, "std": z.scale}
        results.update(hand_parms)

        return results

    def compute_loss(self, prd, tgt, **kwargs):
        device = tgt[Queries.HAND_VERTS_OBJ].device

        if self.v_weights.device != device:
            self.v_weights = self.v_weights.to(device)
            self.v_weights2 = self.v_weights2.to(device)
            self.vpe = self.vpe.to(device)

        dtype = tgt[Queries.HAND_VERTS_OBJ].dtype
        batch_size = tgt[Queries.HAND_VERTS_OBJ].shape[0]

        q_z = torch.distributions.normal.Normal(prd[f"mean"], prd["std"])
        mano_pose = torch.cat([prd["global_orient"], prd["hand_pose"]], dim=1)  # (B, 48)
        mano_shape = tgt[Queries.HAND_SHAPE]  # (B, 10)
        mano_output: MANOOutput = self.mano_layer(mano_pose, mano_shape)

        verts_rhand = mano_output.verts + prd[f"transl"].unsqueeze(1)
        rh_f = self.mano_layer.th_faces.expand(batch_size, -1, -1)

        rh_mesh = Meshes(verts=verts_rhand, faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=tgt[Queries.HAND_VERTS_OBJ],
                            faces=rh_f).to(device).verts_normals_packed().view(-1, 778, 3)

        obj_verts = tgt[Queries.OBJ_VERTS_OBJ_DS]
        n_obj_verts = obj_verts.shape[1]

        o2h_signed, h2o, _ = point2point_signed(verts_rhand, obj_verts, rh_mesh)
        o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(tgt[Queries.HAND_VERTS_OBJ], obj_verts, rh_mesh_gt)
        # addaptive weight for penetration and contact verts
        w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.
        w = torch.ones([batch_size, n_obj_verts]).to(device)
        w[~w_dist] = .1  # less weight for far away vertices
        w[w_dist_neg] = 1.5  # more weight for penetration
        ######### dist loss
        loss_dist_h = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2))
        loss_dist_o = 30 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ij,ij->ij', torch.abs(o2h_signed - o2h_signed_gt), w))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.kl_coef) * torch.mean(
            torch.einsum('ijk,j->ijk', torch.abs(verts_rhand - tgt[Queries.HAND_VERTS_OBJ]), self.v_weights))
        ########## edge loss
        loss_edge = 30 * (1. - self.kl_coef) * self.LossL1(self._edges_for(verts_rhand, self.vpe),
                                                           self._edges_for(tgt[Queries.HAND_VERTS_OBJ], self.vpe))
        ########## KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
        )
        loss_kl = self.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        ##########

        loss_dict = {
            'loss_kl_c': loss_kl,
            'loss_edge_c': loss_edge,
            'loss_mesh_rec_c': loss_mesh_rec_w,
            'loss_dist_h_c': loss_dist_h,
            'loss_dist_o_c': loss_dist_o,
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total_c'] = loss_total
        loss_dict['loss'] = loss_total
        return loss_dict

    def _edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def sample_poses(self, bps_object, seed=None):
        bs = bps_object.shape[0]

        if seed is not None:
            np_rand_state = np.random.get_state()
            np.random.seed(seed)

        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0.0, 1.0, size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen, dtype=dtype).to(device)

        if seed is not None:
            np.random.set_state(np_rand_state)

        result = self.decode(Zgen, bps_object)
        return result
