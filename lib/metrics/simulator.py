import contextlib
import io
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from lib.utils.misc import RedirectStream
from lib.utils.transform import rotmat_to_quat


def take_picture(renderer, width=256, height=256, scale=0.001, conn_id=None):
    import pybullet as p

    view_matrix = p.computeViewMatrix([0, 0, -1], [0, 0, 0], [0, -1, 0], physicsClientId=conn_id)
    proj_matrix = p.computeProjectionMatrixFOV(20, 1, 0.05, 2, physicsClientId=conn_id)
    w, h, rgba, depth, mask = p.getCameraImage(
        width=width,
        height=height,
        projectionMatrix=proj_matrix,
        viewMatrix=view_matrix,
        renderer=renderer,
        physicsClientId=conn_id,
    )
    return rgba


def write_video(frames, path):
    import skvideo.io as skvio
    skvio.vwrite(path, np.array(frames).astype(np.uint8))


def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def vhacd(
    filename,
    resolution=1000,
    concavity=0.001,
    planeDownsampling=4,
    convexhullDownsampling=4,
    alpha=0.05,
    beta=0.0,
    maxhulls=1024,
    pca=0,
    mode=0,
    maxNumVerticesPerCH=64,
    minVolumePerCH=0.0001,
):
    import pybullet as p
    p.vhacd(
        fileNameIn=filename,
        fileNameOut=filename,
        fileNameLogging="/dev/null",
        concavity=concavity,
        alpha=alpha,
        beta=beta,
        minVolumePerCH=minVolumePerCH,
        resolution=resolution,
        maxNumVerticesPerCH=maxNumVerticesPerCH,
        planeDownsampling=planeDownsampling,
        convexhullDownsampling=convexhullDownsampling,
        pca=pca,
        mode=mode,
    )
    return True


def run_simulation(
    hand_verts,
    hand_faces,
    obj_verts,
    obj_faces,
    obj_vhacd_fname,  # path to vhacd
    obj_rotmat,
    conn_id,
    simulation_step=1 / 240,
    num_iterations=35,
    object_friction=3,
    hand_friction=3,
    hand_restitution=0,
    object_restitution=0.5,
    object_mass=1,
    wait_time=0,
    save_video=True,
    save_video_path=None,
    save_hand_path=None,
    save_simul_folder=None,
    base_tmp_dir=None,
    use_gui=False,
):
    import pybullet as p

    hand_indicies = hand_faces.flatten().tolist()
    p.resetSimulation(physicsClientId=conn_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(fixedTimeStep=simulation_step, physicsClientId=conn_id)
    p.setGravity(0, 9.8, 0, physicsClientId=conn_id)

    # add hand
    if base_tmp_dir is None:
        base_tmp_dir = "tmp/simulation/objs"
    os.makedirs(base_tmp_dir, exist_ok=True)
    hand_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
    save_obj(hand_tmp_fname, hand_verts, hand_faces)

    if save_hand_path is not None:
        shutil.copy(hand_tmp_fname, save_hand_path)

    hand_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        indices=hand_indicies,
        physicsClientId=conn_id,
    )
    hand_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        rgbaColor=[0, 0, 1, 1],
        specularColor=[0, 0, 1],
        physicsClientId=conn_id,
    )

    hand_body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=hand_collision_id,
        baseVisualShapeIndex=hand_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        hand_body_id,
        -1,
        lateralFriction=hand_friction,
        restitution=hand_restitution,
        physicsClientId=conn_id,
    )

    # add object
    obj_verts = np.copy(obj_verts)
    obj_center_mass = np.mean(obj_verts, axis=0)
    obj_center_mass_can = obj_rotmat.T @ obj_center_mass
    #obj_verts -= obj_center_mass

    obj_collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_vhacd_fname, physicsClientId=conn_id)
    obj_quat = rotmat_to_quat(obj_rotmat)
    obj_quat_xyzw = obj_quat[..., (1, 2, 3, 0)]

    obj_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_vhacd_fname,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[1, 0, 0],
        physicsClientId=conn_id,
    )
    obj_body_id = p.createMultiBody(
        baseMass=object_mass,
        basePosition=[0, 0, 0],
        baseOrientation=obj_quat_xyzw,
        baseInertialFramePosition=obj_center_mass_can,
        baseInertialFrameOrientation=[0, 0, 0, 1],
        baseCollisionShapeIndex=obj_collision_id,
        baseVisualShapeIndex=obj_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        obj_body_id,
        -1,
        lateralFriction=object_friction,
        restitution=object_restitution,
        physicsClientId=conn_id,
    )

    # simulate for several steps
    if save_video:
        images = []
        if use_gui:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER

    for step_idx in range(num_iterations):
        p.stepSimulation(physicsClientId=conn_id)
        if save_video:
            img = take_picture(renderer, conn_id=conn_id)
            images.append(img)
        if save_simul_folder:
            hand_step_path = os.path.join(save_simul_folder, "{:08d}_hand.obj".format(step_idx))
            shutil.copy(hand_tmp_fname, hand_step_path)
            obj_step_path = os.path.join(save_simul_folder, "{:08d}_obj.obj".format(step_idx))
            pos, orn = p.getBasePositionAndOrientation(obj_body_id, physicsClientId=conn_id)
            mat = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            obj_verts_t = pos + np.dot(mat, obj_verts.T).T
            save_obj(obj_step_path, obj_verts_t, obj_faces)
        time.sleep(wait_time)

    if save_video:
        write_video(images, save_video_path)
        print("Saved gif to {}".format(save_video_path))
    pos_end = p.getBasePositionAndOrientation(obj_body_id, physicsClientId=conn_id)[0]

    os.remove(hand_tmp_fname)
    # distance = np.linalg.norm(pos_end - obj_center_mass)
    distance = np.linalg.norm(pos_end)
    p.disconnect(physicsClientId=conn_id)
    return distance


def simulation_sample(
    sample_idx,
    sample_info,
    save_gif_folder=None,
    save_obj_folder=None,
    tmp_folder=None,
    use_gui=False,
    wait_time=0,
    sample_vis_freq=10,
):
    with RedirectStream(stream=sys.stdout), RedirectStream(
            stream=sys.stderr), io.StringIO() as fstdout, contextlib.redirect_stdout(fstdout):
        import pybullet as p

        if use_gui:
            conn_id = p.connect(p.GUI)
        else:
            conn_id = p.connect(p.DIRECT)
        if sample_idx % sample_vis_freq == 0:
            save_video = True
            if "sample_id" not in sample_info:
                sample_id = "{:04d}".format(sample_idx)
            else:
                sample_id = sample_info["sample_id"]
            save_video_path = os.path.join(save_gif_folder, f"{sample_id}.gif")
            save_obj_path = os.path.join(save_obj_folder, f"{sample_id}_obj.obj")
            save_hand_path = os.path.join(save_obj_folder, f"{sample_id}_hand.obj")
            os.makedirs(os.path.dirname(save_obj_path), exist_ok=True)
            os.makedirs(os.path.dirname(save_hand_path), exist_ok=True)
            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        else:
            save_video = False
            save_video_path = None
            save_obj_path = None
            save_hand_path = None

        distance = run_simulation(
            hand_verts=sample_info["hand_verts"],
            hand_faces=sample_info["hand_faces"],
            obj_verts=sample_info["obj_verts"],
            obj_faces=sample_info["obj_faces"],
            obj_vhacd_fname=sample_info["obj_vhacd_fname"],
            obj_rotmat=sample_info["obj_rotmat"],
            conn_id=conn_id,
            simulation_step=1 / 240,
            object_friction=3,
            hand_friction=3,
            hand_restitution=0,
            object_restitution=0.5,
            object_mass=1,
            wait_time=wait_time,
            save_video=save_video,
            save_hand_path=save_hand_path,
            save_video_path=save_video_path,
            base_tmp_dir=tmp_folder,
            use_gui=use_gui,
        )
    return distance
