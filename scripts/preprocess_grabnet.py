import argparse
import os
import pickle

import numpy as np
import pytorch3d.transforms as torch3d
import torch
import trimesh
from lib.utils.transform import rotmat_to_aa, aa_to_rotmat
from lib.datasets.grabnet_data import GrabNetGrasp, _GrabNetReload
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger

from lib.utils.transform import center_vert_bbox, mano_to_openpose
from termcolor import colored, cprint
from trimesh import Trimesh


def prepare_grabnet_split(split, obj_warehouse, dump_root_dir, no_dump=False):
    from copy import deepcopy

    dump_subset_dir = os.path.join(dump_root_dir, split)
    os.makedirs(dump_subset_dir, exist_ok=True)

    grabnet = _GrabNetReload(
        grabnet_dir="data/GrabNet",
        grab_dir="data/GRAB",
        ds_name=split,
    )
    cprint(f"Loaded GrabNetOfficial {split}, total {len(grabnet)}", "yellow")

    # tailor = Tailor(grabnet.mano_layer)
    mano_layer = grabnet.mano_layer

    for obj_id in grabnet.obj_names:
        grabnet_grasp_list = []
        grasp_annot_path = os.path.join(dump_subset_dir, f"{split}_{obj_id}_grasp_annot.pth")
        obj_frames_ids = grabnet.obj_name2_id_mapping[obj_id]

        # region ===== preprocess object mesh >>>>>
        if obj_id not in obj_warehouse:
            obj_mesh: Trimesh = deepcopy(grabnet.obj_name2_mesh_mapping[obj_id])

            obj_verts_centered, center_shift, _ = center_vert_bbox(np.asfarray(obj_mesh.vertices, dtype=np.float32),
                                                                   scale=False)
            assert np.sum(np.abs(center_shift)) == 0  # GrabNet: the ContactDB object meshes is already centered !
            obj_holder = {
                "mesh_obj": trimesh.Trimesh(vertices=obj_verts_centered, faces=obj_mesh.faces, process=False),
                "verts_obj": obj_verts_centered.astype(np.float32),  # V, in object canonical space
                "faces": obj_mesh.faces.astype(np.int32),  # F, paired with V
                "center_shift": center_shift.astype(np.float32),
            }
            obj_warehouse[obj_id] = obj_holder
            logger.info(f"Finished object: {colored(obj_id, 'blue', attrs=['bold'])} processing.")
        else:
            obj_holder = obj_warehouse[obj_id]
        # endregion

        bar = etqdm(obj_frames_ids)
        bar.set_description(f"GrabNet split:{colored(split, 'green')} object:{colored(obj_id, 'blue')}")
        for _, sid in enumerate(bar):
            sid = int(sid)  # np.int64 to int
            sbj_id: int = grabnet.frame_sbjs.numpy()[sid]
            sbj_name: str = grabnet.sbjs[sbj_id]
            ### Acquire the frame_data
            blob = grabnet[sid]
            frame_data = {k: blob[k] for k in blob.keys()}

            # NOTE: what is in frame_data:
            """
            global_orient_rhand_rotmat torch.Size([1, 3, 3])
            fpose_rhand_rotmat torch.Size([15, 3, 3])
            trans_rhand torch.Size([3])
            trans_obj torch.Size([3])  # ALWAYS ZERO
            root_orient_obj_rotmat torch.Size([1, 3, 3])
            global_orient_rhand_rotmat_f torch.Size([1, 3, 3])
            fpose_rhand_rotmat_f torch.Size([15, 3, 3])
            trans_rhand_f torch.Size([3])
            verts_object torch.Size([2048, 3])
            verts_rhand torch.Size([778, 3])
            verts_rhand_f torch.Size([778, 3])
            bps_object torch.Size([4096])
            """

            # retrive hand verts, joints
            # hand_verts = frame_data["verts_rhand"].numpy()  # TENSOR (778, 3)
            _hand_transl = frame_data["trans_rhand"]  # TENSOR(3,)

            hand_rel_rotmat = frame_data["fpose_rhand_rotmat"]  # (15, 3, 3)
            hand_glob_rotmat = frame_data["global_orient_rhand_rotmat"]  # (1, 3, 3)
            hand_rotmat = torch.cat([hand_glob_rotmat, hand_rel_rotmat], dim=0)  # TENSOR (16, 3, 3)
            hand_pose = rotmat_to_aa(hand_rotmat).reshape(-1)  # TENSOR (16x3)
            beta = grabnet.sbj_betas[sbj_id]  # TENSOR (10)
            mano_out = mano_layer(hand_pose.unsqueeze(0), beta.unsqueeze(0))  # TENSOR (778, 3)
            hand_verts = mano_out.verts + _hand_transl[None, None, :]  # (1, 778, 3)
            hand_joints = mano_to_openpose(mano_layer.th_J_regressor, hand_verts)  # (1, 21, 3)

            # to numpy
            hand_verts = hand_verts.numpy().reshape(-1, 3)  # (778, 3)
            hand_joints = hand_joints.numpy().reshape(-1, 3)  # (21, 3)
            hand_rel_rotmat = hand_rel_rotmat.numpy()  # (15, 3, 3)
            hand_glob_rotmat = hand_glob_rotmat.numpy()  # (1, 3, 3)
            beta = beta.numpy()  # (10,)

            # apply the inverse rotation of object to the hand pose, hand verts and hand
            rot_mat_obj = frame_data["root_orient_obj_rotmat"].numpy().reshape(3, 3).T
            rot_mat_obj_inv = np.linalg.inv(rot_mat_obj)
            transl_obj = frame_data["trans_obj"].numpy()
            assert np.sum(np.abs(transl_obj)) == 0, "GrabNet: something wrong with the object translation !"
            # obj_verts = (rot_mat_obj @ obj_holder["verts_obj"].T).T + transl

            new_hand_verts = (rot_mat_obj_inv @ hand_verts.T).T
            new_hand_joints = (rot_mat_obj_inv @ hand_joints.T).T
            new_hand_glob_rotmat = (rot_mat_obj_inv @ hand_glob_rotmat.reshape(3, 3))[None, :, :]  # (1, 3, 3)
            new_hand_rotmat = np.concatenate([new_hand_glob_rotmat, hand_rel_rotmat], axis=0)  # (16, 3, 3)
            new_hand_pose = rotmat_to_aa(new_hand_rotmat).reshape(-1)  # (16x3)

            # NOTE: dealing with the key that has _f for ''fake'' data to train RefineNet
            hand_rel_rotmat_f = frame_data["fpose_rhand_rotmat_f"]  # (15, 3, 3)
            hand_glob_rotmat_f = frame_data["global_orient_rhand_rotmat_f"]  # (1, 3, 3)
            hand_rotmat_f = torch.cat([hand_glob_rotmat_f, hand_rel_rotmat_f], dim=0)  # TENSOR (16, 3, 3)
            hand_pose_f = torch3d.so3_log_map(hand_rotmat_f).reshape(-1).numpy()  # TENSOR (16x3)
            hand_transl_f = frame_data["trans_rhand_f"].numpy()  # TENSOR(3,)

            grabnet_grasp = GrabNetGrasp(
                split=split,
                sample_idx=sid,
                obj_id=obj_id,
                sbj_name=sbj_name,
                obj_rot=rot_mat_obj.astype(np.float32),
                obj_transl=np.zeros(3).astype(np.float32),
                joints_obj=new_hand_joints.astype(np.float32),
                hand_pose_obj=new_hand_pose.astype(np.float32),
                hand_shape=beta.astype(np.float32),
            )

            # region ===== visualize >>>>>
            """
            import open3d as o3d
            o3d_obj_mesh = o3d.geometry.TriangleMesh()
            o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(obj_holder["faces"])
            o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(obj_holder["verts_obj"])
            o3d_obj_mesh.compute_vertex_normals()

            o3d_hand_mesh = o3d.geometry.TriangleMesh()
            o3d_hand_mesh.triangles = o3d.utility.Vector3iVector(grabnet.mano_layer.th_faces)
            o3d_hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
            o3d_hand_mesh.compute_vertex_normals()

            o3d_hand_mesh2 = o3d.geometry.TriangleMesh()

            o3d_hand_mesh2.triangles = o3d.utility.Vector3iVector(grabnet.mano_layer.th_faces)
            o3d_hand_mesh2.vertices = o3d.utility.Vector3dVector(hand_verts_2 - _hand_transl + new_hand_transl)
            vc = NAME_2_RGB["YellowGreen"][None, :].repeat([778], axis=0)
            o3d_hand_mesh2.vertex_colors = o3d.utility.Vector3dVector(vc)
            o3d_hand_mesh2.compute_vertex_normals()

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name="Runtime HAND + OBJ",
                width=1024,
                height=768,
            )
            o3d.visualization.draw_geometries([
                o3d_obj_mesh,
                o3d_hand_mesh,
                o3d_hand_mesh2,
            ])
            """
            # endregion
            grabnet_grasp_list.append(grabnet_grasp._asdict())

        if no_dump:
            continue

        with open(grasp_annot_path, "wb") as f:
            pickle.dump(grabnet_grasp_list, f)
            logger.info(f"Wrote GrabNet split:{split} object:{obj_id} all frames' annotation in {grasp_annot_path}.")


def prepare_grabnet(args):
    data_root = "data"
    name = "GrabNet"
    dump_root_dir = os.path.join(data_root, f"{name}_grasp")  # ./data/GrabNet_grasp/
    if os.path.exists(dump_root_dir) and args.no_dump is False:
        logger.error(f"Destination folder {dump_root_dir} exist! Do not create dataset twice.")
        # raise Exception

    obj_models_dir = os.path.join(dump_root_dir, "object_models")
    os.makedirs(dump_root_dir, exist_ok=True)
    os.makedirs(obj_models_dir, exist_ok=True)

    splits = ["train", "test", "val"]
    obj_warehouse = {}
    for split in splits:
        prepare_grabnet_split(split, obj_warehouse, dump_root_dir, args.no_dump)
        logger.info(f"Finished split: {colored(split, 'yellow', attrs=['bold'])} processing.")

    for obj_id, obj_holder in obj_warehouse.items():
        mesh_obj = obj_holder["mesh_obj"]
        export_res = trimesh.exchange.ply.export_ply(mesh_obj, encoding='binary')
        obj_model_path = os.path.join(obj_models_dir, f"{obj_id}.ply")
        with open(obj_model_path, "wb+") as f:
            f.write(export_res)
        cprint(f"Exported {obj_id} to {obj_model_path}", "green", attrs=["bold"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess GrabNetData")
    parser.add_argument("--no_dump", help="whether to dump prepared data", action="store_true")
    args = parser.parse_args()
    prepare_grabnet(args)
