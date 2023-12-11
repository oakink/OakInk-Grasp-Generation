from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets.grasp_query import Queries
from lib.utils.builder import MODEL
from lib.utils.transform import aa_to_rotmat

from .coarse_net import CoarseNet
from .grabnet_arch import ResBlock, parms_decode


@MODEL.register_module()
class CoarseIntentNetIntentEmbedEncode(CoarseNet):

    def __init__(self, cfg):
        self.n_intents = cfg.DATA_PRESET.N_INTENTS
        super(CoarseIntentNetIntentEmbedEncode, self).__init__(cfg)

    def _build_layers(self):
        in_bps = self.in_bps
        in_pose = self.in_pose
        n_neurons = self.n_neurons
        latentD = self.latentD
        intentD = self.latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + intentD + in_pose)
        self.enc_rb1 = ResBlock(in_bps + intentD + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + intentD + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + intentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + intentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

        self.intent_embed = nn.Embedding(self.n_intents, intentD)  # (3, 16)
        nn.init.uniform_(self.intent_embed.weight.data, 0, 1)

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent):
        bs = bps_object.shape[0]
        X = torch.cat([intent, bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object, intent):
        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)
        X0 = torch.cat([Zin, intent, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        pose = self.dec_pose(X)
        trans = self.dec_trans(X)
        results = parms_decode(pose, trans)
        results["z"] = Zin
        return results

    def sample_poses(self, bps_object, intent, seed=None):
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

        result = self.decode(Zgen, bps_object, intent)
        return result

    def _forward_impl(self, bps_object, trans_rhand, global_orient_rhand_rotmat, intent, **kwargs):
        z = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat, intent)
        z_s = z.rsample()
        hand_parms = self.decode(z_s, bps_object, intent)
        results = {"mean": z.mean, "std": z.scale}
        results.update(hand_parms)
        return results

    def training_step(self, inp: Dict, step_idx, mode="train", **kwargs):
        bps_object = inp[Queries.OBJ_BPS]  # (B, 4096)
        trans_rhand = inp[Queries.HAND_TRANSL_OBJ]  # (B, 3)
        global_hand_pose_obj = inp[Queries.HAND_POSE_OBJ][:, :3]  # (B, 3)
        global_orient_rhand_rotmat = aa_to_rotmat(global_hand_pose_obj).unsqueeze(1)  # (B, 1, 3, 3)
        intent_vec = inp[Queries.INTENT_VEC]  # (B, INTENT_D)
        bs = bps_object.shape[0]

        # intent_vec has shape: (B, 3), the last dim is a one hot vector that indicates the intent ID
        # self.intent_embed.weight has shape: (3, 16), each row is a vector that represents an intent embedding
        # now use the intent_vec to index self.intent_embed
        # intent has shape: (B, 16)
        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)

        res = self._forward_impl(bps_object, trans_rhand, global_orient_rhand_rotmat, intent)
        cnet_loss = self.compute_loss(prd=res, tgt=inp, **kwargs)
        if step_idx % self.log_freq == 0:
            for k, v in cnet_loss.items():
                self.summary.add_scalar(f"{mode}/{k}", v.item(), step_idx)

        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, cnet_loss

    def testing_step(self, inp, step_idx, intent_name=None, **kwargs):
        obj_bps = inp[Queries.OBJ_BPS]
        bs = obj_bps.shape[0]
        # intent_vec = inp[Queries.INTENT_VEC]  # (B, INTENT_D)
        if intent_name is not None:
            intent_id = int(ALL_INTENT[intent_name])
            intent_vec = torch.zeros((bs, self.n_intents), dtype=torch.float32)
            intent_vec[torch.arange(bs), intent_id - 1] = 1.0  # intent id starts from 1
            intent_vec = intent_vec.to(obj_bps.device)
            inp[Queries.INTENT_VEC] = intent_vec  # @OVERWRITE
        else:
            intent_vec = inp[Queries.INTENT_VEC]

        intent = torch.bmm(intent_vec.unsqueeze(1), self.intent_embed.weight.expand(bs, -1, -1)).squeeze(1)
        res = self.sample_poses(obj_bps, intent=intent, seed=kwargs.get("seed", None))

        res["mano_pose"] = torch.cat([res["global_orient"], res["hand_pose"]], dim=1)
        mano_out = self.mano_layer(res["mano_pose"])
        res["hand_verts"] = mano_out.verts + res[f"transl"].unsqueeze(1)  # (B, 778, 3)
        res = {f"{self.stage}.{k}": v for k, v in res.items()}
        return res, {}
