import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.model_abc import ModelABC
from lib.utils.builder import MODEL, build_model
from lib.utils.transform import rotmat_to_aa


@MODEL.register_module()
class GrabNet(ModelABC):

    def __init__(self, cfg):
        super(GrabNet, self).__init__()
        self.cfg = cfg

        self.CNet = build_model(cfg.COARSE_NET, data_preset=cfg.DATA_PRESET)
        self.RNet = build_model(cfg.REFINE_NET, data_preset=cfg.DATA_PRESET)
        self.mano_layer = self.RNet.mano_layer

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq

    def forward(self, inp, step_idx, mode="train", **kwargs):
        if mode == "test":
            return self.testing_step(inp, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inp, step_idx, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def testing_step(self, inp, step_idx, **kwargs):
        cnet_res, _ = self.CNet(inp=inp, mode='test', step_idx=step_idx, **kwargs)
        rnet_res, _ = self.RNet(inp=inp, cnet_res=cnet_res, mode='test', step_idx=step_idx, **kwargs)
        grabnet_res = {**cnet_res, **rnet_res}

        if kwargs.get("callback") is not None and callable(kwargs["callback"]):
            kwargs["callback"](prd=grabnet_res, inp=inp, step_idx=step_idx, **kwargs)

        evals = {}
        return grabnet_res, evals

    def inference_step(self, inp, step_idx, **kwargs):
        prd, _ = self.testing_step(inp, step_idx, **kwargs)
        return prd

    def compute_loss(self, prd, gts, **kwargs):
        raise NotImplementedError()


class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)

        return Xout


def parms_decode(pose, trans):
    bs = trans.shape[0]
    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 3, 3])
    pose = rotmat_to_aa(pose).view(bs, -1)
    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])
    hand_parms = {"global_orient": global_orient, "hand_pose": hand_pose, "transl": trans, "fullpose": pose_full}
    return hand_parms


def CRot2rotmat(pose):
    reshaped_input = pose.view(-1, 3, 2)
    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)
