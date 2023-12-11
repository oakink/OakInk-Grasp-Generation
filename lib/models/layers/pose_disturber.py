import torch
from torch import nn
from torch.distributions.normal import Normal


class PoseDisturber(nn.Module):

    def __init__(self, tsl_sigma=0.02, pose_sigma=0.2, root_rot_sigma=0.004, **kwargs):
        super().__init__()

        self.hand_transl_dist = Normal(torch.tensor(0.0), tsl_sigma)
        self.hand_pose_dist = Normal(torch.tensor(0.0), pose_sigma)
        self.hand_root_rot_dist = Normal(torch.tensor(0.0), root_rot_sigma)

    def forward(self, hand_pose, hand_transl):
        batch_size = hand_pose.shape[0]
        device = hand_pose.device

        hand_root_pose = hand_pose[:, :3]
        hand_rel_pose = hand_pose[:, 3:]

        hand_transl = hand_transl + self.hand_transl_dist.sample((batch_size, 3)).to(device)
        hand_root_pose = hand_root_pose + self.hand_root_rot_dist.sample((batch_size, 3)).to(device)
        hand_rel_pose = hand_rel_pose + self.hand_pose_dist.sample((batch_size, 15 * 3)).to(device)
        hand_pose = torch.cat([hand_root_pose, hand_rel_pose], dim=1)

        return hand_pose, hand_transl
