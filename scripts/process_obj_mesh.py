import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys

import trimesh
from joblib import Parallel, delayed
from oikit.oi_shape import OakInkShape
from tqdm import tqdm

from lib.utils.misc import RedirectStream


def watertight_one(oid, omesh, export_dir):
    # export and skip materials
    omesh.visual = trimesh.visual.ColorVisuals()  # https://github.com/mikedh/trimesh/issues/1219
    temp_path = os.path.join(export_dir, f"_{oid}.obj")
    trimesh.exchange.export.export_mesh(omesh, temp_path, file_type="obj")

    out_path = os.path.join(export_dir, f"{oid}.obj")
    try:
        depth_list = [8, 7, 6]
        for d in depth_list:
            command = [
                "thirdparty/ManifoldPlus/build/manifold", "--input", temp_path, "--output", out_path, "--depth",
                str(d)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            obj_reload = trimesh.load(out_path, process=False, force="mesh", skip_materials=True)
            assert obj_reload.is_watertight, f"{out_path} is not watertight"
            if obj_reload.faces.shape[0] < 30000:
                break
        # remove temp file
        os.remove(temp_path)
    except Exception as e:
        # dump the error message to a file
        with open(os.path.join(export_dir, f"{oid}.err"), "w") as f:
            f.write(str(e))


def watertight(args):
    oishape = OakInkShape(category="all",
                          intent_mode="all",
                          data_split="all",
                          use_cache=True,
                          use_downsample_mesh=True,
                          preload_obj=True)

    export_dir = os.path.join(args.proc_dir, "watertight")
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    for oid, omesh in tqdm(oishape.obj_warehouse.items()):
        tasks.append(delayed(watertight_one)(oid, omesh, export_dir))

    Parallel(n_jobs=args.n_jobs, verbose=10, timeout=60)(tasks)
    return


def voxel_one(oid, input_mesh_path, export_dir):
    command = ["thirdparty/binvox", "-d", "128", input_mesh_path]
    subprocess.run(command, stdout=subprocess.DEVNULL)
    res_binvox_path = input_mesh_path.replace(".obj", ".binvox")
    out_mesh_dest = os.path.join(export_dir, f"{oid}.binvox")
    shutil.move(res_binvox_path, out_mesh_dest)


def voxel(args):
    export_dir = os.path.join(args.proc_dir, "voxel")
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    all_wt_obj_flist = os.listdir(os.path.join(args.proc_dir, "watertight"))
    for wt_obj_f in all_wt_obj_flist:
        oid, suffix = wt_obj_f.split(".")[0], wt_obj_f.split(".")[1]
        input_mesh_path = os.path.join(args.proc_dir, "watertight", wt_obj_f)
        tasks.append(delayed(voxel_one)(oid, input_mesh_path, export_dir))

    Parallel(n_jobs=args.n_jobs, verbose=10, timeout=60)(tasks)
    return


def vhacd_one(oid, input_mesh_path, export_dir):
    res_vhacd_path = os.path.join(export_dir, f"{oid}.obj")
    with RedirectStream(stream=sys.stdout), RedirectStream(stream=sys.stderr), io.StringIO() as \
        fstdout, contextlib.redirect_stdout(fstdout):
        import pybullet as pbl
        pbl.vhacd(
            fileNameIn=input_mesh_path,
            fileNameOut=res_vhacd_path,
            fileNameLogging="/dev/null",
            resolution=100000,
            concavity=0.0025,
            planeDownsampling=4,
            convexhullDownsampling=4,
            alpha=0.05,
            beta=0.0,
            pca=0,
            mode=0,
            maxNumVerticesPerCH=64,
            minVolumePerCH=0.0001,
        )


def vhacd(args):
    export_dir = os.path.join(args.proc_dir, "vhacd")
    os.makedirs(export_dir, exist_ok=True)

    all_wt_obj_flist = os.listdir(os.path.join(args.proc_dir, "watertight"))

    tasks = []
    for wt_obj_f in all_wt_obj_flist:
        oid, suffix = wt_obj_f.split(".")[0], wt_obj_f.split(".")[1]
        input_mesh_path = os.path.join(args.proc_dir, "watertight", wt_obj_f)
        tasks.append(delayed(vhacd_one)(oid, input_mesh_path, export_dir))

    Parallel(n_jobs=args.n_jobs, verbose=10, timeout=60)(tasks)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watertight and Voxelize")
    parser.add_argument("--stage", type=str, default="watertight", choices=["watertight", "voxel", "vhacd"])
    parser.add_argument("--proc_dir", type=str, default="data/OakInkShape_object_process/")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    if args.stage == "watertight":
        watertight(args)
    elif args.stage == "voxel":
        voxel(args)
    elif args.stage == "vhacd":
        vhacd(args)
