import numpy as np
import trimesh
from scipy.spatial.distance import cdist


def batch_pairwise_dist(x, y):
    zz = np.einsum("bij,bjk->bik", x, y.transpose(0, 2, 1))
    rx = np.sum(x**2, axis=2)[:, :, None].repeat(zz.shape[-1], axis=-1)
    ry = np.sum(y**2, axis=2)[:, None, :].repeat(zz.shape[-2], axis=-2)
    P = rx + ry - 2 * zz
    return P


def penetration(obj_verts, obj_faces, hand_verts, mode="max"):
    from thirdparty.libmesh.inside_mesh import check_mesh_contains
    obj_trimesh = trimesh.Trimesh(vertices=np.asarray(obj_verts), faces=np.asarray(obj_faces))
    inside = check_mesh_contains(obj_trimesh, hand_verts)

    valid_vals = inside.sum()
    if valid_vals > 0:
        selected_hand_verts = hand_verts[inside, :]

        mins_sel_hand_to_obj = np.min(cdist(selected_hand_verts, obj_verts), axis=1)

        collision_vals = mins_sel_hand_to_obj
        if mode == "max":
            penetr_val = np.max(collision_vals)  # max
        elif mode == "mean":
            penetr_val = np.mean(collision_vals)
        elif mode == "sum":
            penetr_val = np.sum(collision_vals)
        else:
            raise KeyError("unexpected penetration mode")
    else:
        penetr_val = 0
    return penetr_val
