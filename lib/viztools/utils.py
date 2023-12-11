from enum import Enum

import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import torch
from matplotlib.colors import get_named_colors_mapping

ColorsMap = get_named_colors_mapping()


class VizHandMode(Enum):
    HAND = 0
    HAND_F = 1
    HAND_BOTH = 2
    HAND_NONE = 3


class ColorMode():
    VERTEX_CONTACT = "vertex_contact"
    CONTACT_REGION = "contact_region"
    ANCHOR_ELASTI = "anchor_elasti"
    PENETRATION = "penetration"
    CONTACTNESS = "contactness"


def get_color_map(x, mode: ColorMode):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if mode == ColorMode.VERTEX_CONTACT:
        n_verts = x.shape[0]
        x = x.reshape(-1)
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[x == 1] = np.array([160, 0, 0]) / 255.0
        vertex_color[x == 0] = np.array([0, 0, 160]) / 255.0
        return vertex_color
    elif mode == ColorMode.ANCHOR_ELASTI or mode == ColorMode.CONTACTNESS:
        cm = cmx.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=1.0)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        vertex_color = scalarMap.to_rgba(x).squeeze()
        return vertex_color[:, :3]
    elif mode == ColorMode.CONTACT_REGION:
        n_verts = x.shape[0]
        x = x.reshape(-1)
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[x == 0] = np.array([207, 56, 112]) / 255.0
        vertex_color[x == 1] = np.array([226, 53, 74]) / 255.0
        vertex_color[x == 2] = np.array([231, 91, 84]) / 255.0

        vertex_color[x == 3] = np.array([235, 105, 79]) / 255.0
        vertex_color[x == 4] = np.array([230, 109, 91]) / 255.0
        vertex_color[x == 5] = np.array([202, 67, 99]) / 255.0

        vertex_color[x == 6] = np.array([240, 162, 62]) / 255.0
        vertex_color[x == 7] = np.array([244, 192, 99]) / 255.0
        vertex_color[x == 8] = np.array([239, 179, 145]) / 255.0

        vertex_color[x == 9] = np.array([224, 231, 243]) / 255.0
        vertex_color[x == 10] = np.array([175, 186, 242]) / 255.0
        vertex_color[x == 11] = np.array([195, 212, 240]) / 255.0

        vertex_color[x == 12] = np.array([50, 115, 173]) / 255.0
        vertex_color[x == 13] = np.array([82, 148, 200]) / 255.0
        vertex_color[x == 14] = np.array([124, 191, 239]) / 255.0

        vertex_color[x == 15] = np.array([144, 78, 150]) / 255.0
        vertex_color[x == 16] = np.array([40, 76, 121]) / 255.0

        vertex_color[x == 17] = np.array([255, 255, 0]) / 255.0
        return vertex_color
    elif mode == ColorMode.PENETRATION:
        n_verts = x.shape[0]
        x = x.reshape(-1)
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[x <= 0.001] = np.array([255, 0, 0]) / 255.0
        vertex_color[x > 0.001] = np.array([220, 220, 220]) / 255.0
        return vertex_color
    else:
        raise NotImplementedError
