from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from pytorch3d.transforms import (axis_angle_to_matrix, axis_angle_to_quaternion, euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_axis_angle, quaternion_to_matrix, rotation_6d_to_matrix)

from .misc import CONST


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript. 

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = 'xyz', **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, np.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError('Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles f{axis_angle.shape}.')
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def aa_to_rot6d(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def ee_to_rotmat(euler_angle: Union[torch.Tensor, np.ndarray], convention='xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert euler angle to rotation matrixs.
    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler angles shape f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix])
    return t(euler_angle, convention.upper())


def rotmat_to_ee(matrix: Union[torch.Tensor, np.ndarray], convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to euler angle.
    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix shape f{matrix.shape}.')
    t = Compose([matrix_to_euler_angles])
    return t(matrix, convention.upper())


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions shape f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d shape f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def _rotate_smpl_pose(pose, rot):
    """Rotate SMPL pose parameters.

    SMPL (https://smpl.is.tue.mpg.de/) is a 3D human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation rad.
    Returns:
        pose_rotated
    """
    rot_mat = _construct_rotation_matrix(rot)
    pose_rotated = pose.copy()
    orient = pose[:3]
    orient_mat = aa_to_rotmat(orient)

    new_orient_mat = np.matmul(rot_mat, orient_mat)
    new_orient = rotmat_to_aa(new_orient_mat)
    pose_rotated[:3] = new_orient

    return pose_rotated


def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.

    Args:
        rot (float): Rotation rad.
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat


def _transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def _get_affine_transform(center, scale, optical_center, out_res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -optical_center[0]
    t_mat[1, 2] = -optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    post_rot_trans = _get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = _get_affine_trans_no_rot(transformed_center[:2], scale, out_res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32)


def _affine_transform(center, scale, out_res, rot=0):
    rotmat = _construct_rotation_matrix(rot=rot, size=3)
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = (rotmat.dot(np.concatenate([center, np.ones(1)])))[:2]

    post_rot_trans = _get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rotmat)
    return total_trans.astype(np.float32)


def _affine_transform_post_rot(center, scale, optical_center, out_res, rot=0):
    rotmat = _construct_rotation_matrix(rot=rot, size=3)
    t_mat = np.eye(3)
    t_mat[0, 2] = -optical_center[0]
    t_mat[1, 2] = -optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rotmat).dot(t_mat).dot(np.concatenate([center, np.ones(1)]))
    affine_trans_post_rot = _get_affine_trans_no_rot(transformed_center[:2], scale, out_res)

    return affine_trans_post_rot.astype(np.float32)


def _get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    scale_ratio = float(res[0]) / float(res[1])
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale * scale_ratio
    affinet[0, 2] = res[0] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[1] * (-float(center[1]) / scale * scale_ratio + 0.5)
    affinet[2, 2] = 1
    return affinet


def batch_xyz2uvd(xyz: torch.Tensor,
                  root_joint: torch.Tensor,
                  intr: torch.Tensor,
                  inp_res: List[int],
                  depth_range=0.4,
                  ref_bone_len: Optional[torch.Tensor] = None,
                  camera_mode="persp") -> torch.Tensor:

    inp_res = torch.Tensor(inp_res).to(xyz.device)  # TENSOR (2,)
    batch_size = xyz.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(xyz.device)  # TENSOR (B, 1)

    if camera_mode == "persp":
        assert intr.dim() == 3, f"Unexpected dim, expect intr has shape (B, 3, 3), got {intr.shape}"
        #  1. normalize depth : root_relative, scale_invariant
        z = xyz[:, :, 2]  # TENSOR (B, NKP)
        xy = xyz[:, :, :2]  # TENSOR (B, NKP, 2)
        xy_ = xy / z.unsqueeze(-1).expand_as(xy)  # TENSOR (B, NKP, 2)
        root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
        z_ = (z - root_joint_z.expand_as(z)) / ref_bone_len.expand_as(z)  # TENSOR (B, NKP)

        #  2. xy_ -> uv
        fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
        fy = intr[:, 1, 1].unsqueeze(-1)
        cx = intr[:, 0, 2].unsqueeze(-1)
        cy = intr[:, 1, 2].unsqueeze(-1)
        # cat 4 TENSOR (B, 1)
        camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
        camparam = camparam.unsqueeze(1).expand(-1, xyz.shape[1], -1)  # TENSOR (B, NKP, 4)
        uv = (xy_ * camparam[:, :, :2]) + camparam[:, :, 2:4]  # TENSOR (B, NKP, 2)

        #  3. normalize uvd to 0~1
        uv = torch.einsum("bij, j->bij", uv, 1.0 / inp_res)  # TENSOR (B, NKP, 2), [0 ~ 1]
        d = z_ / depth_range + 0.5  # TENSOR (B, NKP), [0 ~ 1]
        uvd = torch.cat((uv, d.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)
    elif camera_mode == "ortho":
        assert intr.dim() == 2, f"Unexpected dim, expect intr has shape (B, 3), got {intr.shape}"
        # root_relative
        xyz = xyz - root_joint.unsqueeze(1)  # TENSOR (B, NKP, 3)

        xy = xyz[:, :, :2]  # TENSOR (B, NKP, 2)
        z = xyz[:, :, 2]  # TENSOR (B, NKP)
        z_ = z / ref_bone_len.expand_as(z)  # TENSOR (B, NKP)
        d = z_ / depth_range + 0.5  # TENSOR (B, NKP), [0 ~ 1]

        scale = intr[:, :1].unsqueeze(1)  # TENSOR (B, 1, 1)
        shift = intr[:, 1:].unsqueeze(1)  # TENSOR (B, 1, 2)
        uv = xy * scale + shift  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
        uv = torch.einsum("bij,j->bij", uv, 1.0 / inp_res)  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
        uvd = torch.cat((uv, d.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)

    return uvd


def batch_uvd2xyz(uvd: torch.Tensor,
                  root_joint: torch.Tensor,
                  intr: torch.Tensor,
                  inp_res: List[int],
                  depth_range: float = 0.4,
                  ref_bone_len: Optional[torch.Tensor] = None,
                  camera_mode="persp"):

    inp_res = torch.Tensor(inp_res).to(uvd.device)
    batch_size = uvd.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(uvd.device)

    #  1. denormalized uvd
    uv = torch.einsum("bij,j->bij", uvd[:, :, :2], inp_res)  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
    d = (uvd[:, :, 2] - 0.5) * depth_range  # TENSOR (B, NKP), [-0.2 ~ 0.2]

    if camera_mode == "persp":
        assert intr.dim() == 3, f"Unexpected dim, expect intr has shape (B, 3, 3), got {intr.shape}"
        root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
        z = d * ref_bone_len + root_joint_z.expand_as(uvd[:, :, 2])  # TENSOR (B, NKP)

        #  2. uvd->xyz
        # camparam = torch.zeros((batch_size, 4)).float().to(uvd.device)  # TENSOR (B, 4)
        fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
        fy = intr[:, 1, 1].unsqueeze(-1)
        cx = intr[:, 0, 2].unsqueeze(-1)
        cy = intr[:, 1, 2].unsqueeze(-1)
        # cat 4 TENSOR (B, 1)
        camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
        camparam = camparam.unsqueeze(1).expand(-1, uvd.shape[1], -1)  # TENSOR (B, NKP, 4)
        xy_ = (uv - camparam[:, :, 2:4]) / camparam[:, :, :2]  # TENSOR (B, NKP, 2)
        xy = xy_ * z.unsqueeze(-1).expand_as(uv)  # TENSOR (B, NKP, 2)
        xyz = torch.cat((xy, z.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)
    elif camera_mode == "ortho":
        assert intr.dim() == 2, f"Unexpected dim, expect intr has shape (B, 3), got {intr.shape}"
        scale = intr[:, :1].unsqueeze(1)  # TENSOR (B, 1, 1)
        shift = intr[:, 1:].unsqueeze(1)  # TENSOR (B, 1, 2)
        xy = (uv - shift) / scale
        z = d * ref_bone_len
        xyz = torch.cat((xy, z.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)

        # add root back
        xyz = xyz + root_joint.unsqueeze(1)  # TENSOR (B, NKP, 3)

    return xyz


def batch_persp_project(verts: torch.Tensor, camintr: torch.Tensor):
    """Batch apply perspective procjection on points

    Args:
        verts (torch.Tensor): 3D points with shape (B, N, 3)
        camintr (torch.Tensor): intrinsic matrix with shape (B, 3, 3)

    Returns:
        torch.Tensor: shape (B, N, 2)
    """
    # Project 3d vertices on image plane
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def persp_project(points3d, cam_intr):
    hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
    points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
    return points2d.astype(np.float32)


def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None, scale=False):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if scale:
        if bbox_scale is None:
            bbox_scale = np.linalg.norm(vertices, 2, 1).max()
        vertices = vertices / bbox_scale
    else:
        bbox_scale = 1
    return vertices, bbox_center, bbox_scale


def mano_to_openpose(J_regressor, mano_verts):
    mano_joints = torch.matmul(J_regressor, mano_verts)
    kpId2vertices = CONST.MANO_KPID_2_VERTICES
    tipsId = [v[0] for k, v in kpId2vertices.items()]
    tips = mano_verts[:, tipsId]
    openpose_joints = torch.cat([mano_joints, tips], dim=1)
    # Reorder joints to match OpenPose definition
    openpose_joints = openpose_joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    return openpose_joints


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array([
        [0, -z_unit_Arr[2], z_unit_Arr[1]],
        [z_unit_Arr[2], 0, -z_unit_Arr[0]],
        [-z_unit_Arr[1], z_unit_Arr[0], 0],
    ])

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array([
        [0, -z_c_vec[2], z_c_vec[1]],
        [z_c_vec[2], 0, -z_c_vec[0]],
        [-z_c_vec[1], z_c_vec[0], 0],
    ])

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat
