import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

from lib.utils.builder import MODEL
from lib.datasets.grasp_query import Queries
from lib.utils.transform import aa_to_rotmat
from .coarse_net import CoarseNet
from .grabnet_arch import ResBlock, parms_decode


@MODEL.register_module()
class CoarseHandoverNet(CoarseNet):

    def __init__(self, cfg):
        super(CoarseHandoverNet, self).__init__(cfg, strict=True)

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose * 2)
        self.enc_rb1 = ResBlock(in_bps + in_pose * 2, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose * 2, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + in_bps + in_pose, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_bps + in_pose, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(
        self,
        bps_object,
        trans_rhand,
        global_orient_rhand_rotmat,
        alt_trans_rhand,
        alt_global_orient_rhand_rotmat,
        **kwargs,
    ):
        bs = bps_object.shape[0]
        X = torch.cat(
            [
                bps_object,
                global_orient_rhand_rotmat.view(bs, -1),
                trans_rhand,
                alt_global_orient_rhand_rotmat.view(bs, -1),
                alt_trans_rhand,
            ],
            dim=1,
        )

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object, alt_trans_rhand, alt_global_orient_rhand_rotmat):
        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat(
            [
                Zin,
                o_bps,
                alt_global_orient_rhand_rotmat.view(bs, -1),
                alt_trans_rhand,
            ],
            dim=1,
        )
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        results = parms_decode(pose, trans)
        results["z"] = Zin

        return results

    def sample_poses(
        self,
        bps_object,
        alt_trans_rhand,
        alt_global_orient_rhand_rotmat,
        seed=None,
    ):
        bs = bps_object.shape[0]
        dtype = bps_object.dtype
        device = bps_object.device

        if seed is not None:
            np_rand_state = np.random.get_state()
            np.random.seed(seed)

        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0.0, 1.0, size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen, dtype=dtype).to(device)

        if seed is not None:
            np.random.set_state(np_rand_state)

        res = self.decode(Zgen, bps_object, alt_trans_rhand, alt_global_orient_rhand_rotmat)
        return res

    def _forward_impl(
        self,
        bps_object,
        trans_rhand,
        global_orient_rhand_rotmat,
        alt_trans_rhand,
        alt_global_orient_rhand_rotmat,
        **kwargs,
    ):

        z = self.encode(
            bps_object,
            trans_rhand,
            global_orient_rhand_rotmat,
            alt_trans_rhand,
            alt_global_orient_rhand_rotmat,
            **kwargs,
        )
        z_s = z.rsample()
        hand_parms = self.decode(z_s, bps_object, alt_trans_rhand, alt_global_orient_rhand_rotmat)
        results = {"mean": z.mean, "std": z.scale}
        results.update(hand_parms)
        return results

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        bs = bps_object.shape[0]
        trans_rhand = inp[Queries.HAND_TRANSL_OBJ]  # (B, 3)
        global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]  # (B, 3)
        fpose_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, 3:]  # (B, 45)
        global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)

        alt_trans_rhand = inp[Queries.ALT_HAND_TRANSL_OBJ]  # (B, 3)
        alt_global_hand_pose_obj = inp[Queries.ALT_HAND_POSE_OBJ][:, :3]  # (B, 3)
        alt_global_orient_rhand_rotmat = aa_to_rotmat(alt_global_hand_pose_obj).unsqueeze(1)

        res = self._forward_impl(bps_object, trans_rhand, global_orient_rhand_rotmat, alt_trans_rhand,
                                 alt_global_orient_rhand_rotmat)
        cnet_loss = self.compute_loss(res, inp, **kwargs)
        if step_idx % self.log_freq == 0:
            for k, v in cnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, cnet_loss

    def compute_loss(self, prd, tgt, **kwargs):
        return super().compute_loss(prd, tgt, **kwargs)

    def testing_step(self, inp, step_idx, **kwargs):
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        bs = bps_object.shape[0]

        alt_trans_rhand = inp[Queries.ALT_HAND_TRANSL_OBJ]  # (B, 3)
        alt_global_hand_pose_obj = inp[Queries.ALT_HAND_POSE_OBJ][:, :3]
        alt_global_orient_rhand_rotmat = aa_to_rotmat(alt_global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)

        res = self.sample_poses(bps_object, alt_trans_rhand, alt_global_orient_rhand_rotmat)

        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res[f"transl"].unsqueeze(1)  # (B, 778, 3)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}
