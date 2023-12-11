import numpy as np
import trimesh


def solid_intersection_volume(hand_verts, hand_faces, obj_vox_points, obj_vox_el_vol, return_kin=False):
    # create hand trimesh
    from thirdparty.libmesh.inside_mesh import check_mesh_contains
    hand_trimesh = trimesh.Trimesh(vertices=np.asarray(hand_verts), faces=np.asarray(hand_faces))

    inside = check_mesh_contains(hand_trimesh, obj_vox_points)
    volume = inside.sum() * obj_vox_el_vol
    if return_kin:
        return volume, inside
    else:
        return volume
