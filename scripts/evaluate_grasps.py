import argparse
import os
import pickle

import numpy as np
import torch
import trimesh
from joblib import Parallel, delayed
from manotorch.utils.anchorutils import anchor_load_driver
from trimesh import Trimesh

from lib.metrics.basic_metric import AverageMeter
from lib.metrics.disjointedness import region_disjointness_metric
from lib.metrics.intersection import solid_intersection_volume
from lib.metrics.penetration import penetration
from lib.metrics.simulator import simulation_sample


class DumpedGraspsLoader(object):

    def __init__(self, dumped_grasps_dir, proc_dir):
        self.dumped_grasps_dir = dumped_grasps_dir
        self.proc_dir = proc_dir
        grasp_fname = sorted(os.listdir(dumped_grasps_dir))
        grasp_fname = [el for el in grasp_fname if el.endswith(".pkl")]
        self.grasp_files = [os.path.join(dumped_grasps_dir, el) for el in grasp_fname]
        hand_mesh_wt = trimesh.load("assets/hand_mesh_watertight.obj", process=False)
        self.hand_wt_faces = np.array(hand_mesh_wt.faces, dtype=np.int32)

    def __len__(self):
        return len(self.grasp_files)

    def __getitem__(self, idx):
        with open(self.grasp_files[idx], "rb") as f:
            grasp_item = pickle.load(f)

        # get the filename(sample_id) of the current grasp_files[idx]:
        sample_id = os.path.basename(self.grasp_files[idx]).split(".")[0]
        # sample_id = {obj_id}_{grasp_id}
        grasp_item["sample_id"] = sample_id

        obj_id = grasp_item["obj_id"]
        obj_wt = trimesh.load(os.path.join(self.proc_dir, "watertight", f"{obj_id}.obj"), process=False)
        obj_vox = trimesh.load(os.path.join(self.proc_dir, "voxel", f"{obj_id}.binvox"))
        obj_vhacd_path = os.path.join(self.proc_dir, "vhacd", f"{obj_id}.obj")

        grasp_item["obj_wt"] = obj_wt
        grasp_item["obj_vox"] = obj_vox
        grasp_item["obj_vhacd_path"] = obj_vhacd_path
        grasp_item["hand_faces"] = self.hand_wt_faces
        return grasp_item


def evaluate_grasps(idx, grasp_loader, palm_vert_assignment, palm_vert_idx, sims_dir):
    grasp_item = grasp_loader[idx]
    sample_id = grasp_item["sample_id"]  #
    obj_wt: Trimesh = grasp_item["obj_wt"]
    obj_vox: Trimesh = grasp_item["obj_vox"]
    obj_vhacd_path: str = grasp_item["obj_vhacd_path"]
    obj_rotmat = grasp_item["obj_rotmat"]
    hand_verts = grasp_item["hand_verts_r"]
    hand_faces = grasp_item["hand_faces"]

    obj_wt_verts = np.asarray(obj_wt.vertices, dtype=np.float32)
    obj_wt_faces = np.array(obj_wt.faces, dtype=np.int32)
    obj_vox_points = np.asfarray(obj_vox.points, dtype=np.float32)
    obj_element_volume = obj_vox.element_volume

    # penetration
    pentr_dep = penetration(obj_verts=obj_wt_verts, obj_faces=obj_wt_faces, hand_verts=hand_verts)

    # solid intersection volume
    pentr_vol = solid_intersection_volume(hand_verts=hand_verts,
                                          hand_faces=hand_faces,
                                          obj_vox_points=obj_vox_points,
                                          obj_vox_el_vol=obj_element_volume,
                                          return_kin=False)

    # disjoint distance
    _, disjo_dist = region_disjointness_metric(hand_verts=hand_verts,
                                               obj_verts=obj_wt_verts,
                                               hand_region_assignment=palm_vert_assignment)

    # simulation displacement
    sims_disp = simulation_sample(sample_idx=idx,
                                  sample_info={
                                      "sample_id": sample_id,
                                      "hand_verts": hand_verts,
                                      "hand_faces": hand_faces,
                                      "obj_verts": obj_wt_verts,
                                      "obj_faces": obj_wt_faces,
                                      "obj_vhacd_fname": obj_vhacd_path,
                                      "obj_rotmat": obj_rotmat
                                  },
                                  save_gif_folder=os.path.join(sims_dir, "gif"),
                                  save_obj_folder=os.path.join(sims_dir, "vhacd"),
                                  tmp_folder=os.path.join(sims_dir, "tmp"),
                                  use_gui=False,
                                  sample_vis_freq=1)

    eval_res = {
        "sample_id": sample_id,
        "pentr_dep": pentr_dep,
        "pentr_vol": pentr_vol,
        "disjo_dist": disjo_dist,
        "sims_disp": sims_disp,
    }
    return eval_res


def main(arg):
    dumped_grasps_dir = os.path.join(arg.exp_path, "results")
    simulation_dir = os.path.join(arg.exp_path, "simulation")
    evaluation_dir = os.path.join(arg.exp_path, "evaluations")
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    dumped_grasps_loader = DumpedGraspsLoader(dumped_grasps_dir, arg.proc_dir)
    merged_vertex_assignment = anchor_load_driver("./assets")[2]  # vertex that belongs to a certain palm region
    hand_palm_vert_idx = np.loadtxt("assets/hand_palm_full.txt", dtype=np.int32)

    task_list = []
    print(f"Total number of grasps: {len(dumped_grasps_loader)} to be evaluated.")
    for i in range(len(dumped_grasps_loader)):
        task_list.append(
            delayed(evaluate_grasps)(i, dumped_grasps_loader, merged_vertex_assignment, hand_palm_vert_idx,
                                     simulation_dir))

    pentr_dep = AverageMeter("pentr_dep")
    pentr_vol = AverageMeter("pentr_vol")
    disjo_dist = AverageMeter("disjo_dist")
    sims_disp = AverageMeter("sims_disp")
    sims_disp_list = []  # for calculating std

    eval_res = Parallel(n_jobs=arg.n_jobs, verbose=10)(task_list)
    # dump eval_res
    with open(os.path.join(evaluation_dir, "eval_res.pkl"), "wb") as f:
        pickle.dump(eval_res, f)

    for i, sample_res in enumerate(eval_res):
        pentr_dep.update(sample_res["pentr_dep"])
        pentr_vol.update(sample_res["pentr_vol"])
        disjo_dist.update(sample_res["disjo_dist"])
        sims_disp.update(sample_res["sims_disp"])
        sims_disp_list.append(sample_res["sims_disp"])

    with open(os.path.join(evaluation_dir, "Metric.txt"), "a") as f:
        f.write(f"pentr_dep mean: {pentr_dep.avg}\n")
        f.write(f"pentr_vol mean: {pentr_vol.avg}\n")
        f.write(f"disjo_dist mean: {disjo_dist.avg}\n")
        f.write(f"sims_disp mean: {sims_disp.avg}\n")
        f.write(f"sims_disp std: {np.std(sims_disp_list)}\n")

    print(f"Evaluation done!, results are saved in: {evaluation_dir}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval_dump')
    parser.add_argument("-g", "--gpu_id", type=str, default=0, help="override enviroment var CUDA_VISIBLE_DEVICES")
    parser.add_argument("--n_jobs", type=int, default=8, help="number of jobs for parallel processing")
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--proc_dir",
                        type=str,
                        default="data/OakInkShape_object_process",
                        help="directory to save the processed object meshes")
    arg = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()

    main(arg)
