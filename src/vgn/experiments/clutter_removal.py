import collections
from datetime import datetime
import uuid
import open3d as o3d

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pandas as pd
import pybullet
import tqdm
from IPython import embed
import trimesh
from scipy.spatial.transform import Rotation as R

from vgn.utils.transform import Transform

from vgn import io#, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim, TSDFVolume
from vgn.utils.transform import Rotation, Transform
import pyrender
import matplotlib.pyplot as plt
from vgn.perception import *

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    rviz=False,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed)
    logger = Logger(logdir, description)

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)
            # embed()
            if pc.is_empty():
                break  # empty point cloud, abort this round TODO this should not happen

            # visualize scene
            # if rviz:
            #     vis.clear()
            #     vis.draw_workspace(sim.size)
            #     vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
            #     vis.draw_points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, timings["planning"] = grasp_plan_fn(state)
            if len(grasps) == 0:
                break  # no detections found, abort this round

            # if rviz:
            #     vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            # execute grasp
            grasp, score = grasps[0], scores[0]
            # p1 = grasps[0].pose * Transform(Rotation.identity(), [0.0, -grasps[0].width / 2, 0.05 / 2])
            # p2 = grasps[0].pose * Transform(Rotation.identity(), [0.0, grasps[0].width / 2, 0.05 / 2])
            # if rviz:
            #     vis.draw_grasp(grasp, score, sim.gripper.finger_depth)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label

def compute_img(mesh, camera_pose, camera):
    # pose = ((0.09219822079224008, 0.19074373673839115, 0.08902799005464454),
    #         (-0.29992684865556185,
    #          -0.5278026398047863,
    #          0.6908860912057255,
    #          0.39261261804623226))

    # rot = pybullet.getMatrixFromQuaternion(pose[1])
    # trans = np.array(pose[0])
    # tranform = np.eye(4)
    # tranform[:3, :3] = np.array(rot).reshape(3, 3)
    # tranform[:3, 3] = trans

    # mesh = pyrender.Mesh.from_trimesh(mesh, poses=tranform)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # mesh = pyrender.Mesh.from_points(np.random.rand(100,3))

    scene = pyrender.Scene()
    scene.add(mesh)
    # print(mesh)
    # embed()

    # fx, fy, cx, cy = 540.0, 540.0, 320.0, 240.0
    # camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)

    # camera_ex = np.array([[0., 1., 0., -0.15],
    #                       [0.8660254, 0., -0.5, -0.12990381],
    #                       [-0.5, 0., -0.8660254, 0.675],
    #                       [0., 0., 0., 1]])

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle = np.pi / 16.0, outerConeAngle = np.pi / 6.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # plt.savefig('./evaluation/tmp.png')

    return depth

def run_baseline(
    grasp_plan_fn,
    meshes,
    intrinsic,
    camera_in,
):
    rotX = np.eye(4)
    rotX[0:3, 0:3] = R.from_euler('x', 180, degrees=True).as_matrix()
    
    # scan the scene
    mesh = trimesh.util.concatenate(meshes)
    # for i in meshes:
    #     i.show()
    # mesh.show()

    size = 0.3 # voxel_size * resolution
    tsdf = TSDFVolume(size, 40)
    tsdf_high = TSDFVolume(size, 120)
    
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0])
    r = 2.0 * size
    theta = np.pi / 6.0

    n = 6
    N = n
    phi_list = 2.0 * np.pi * np.arange(n) / N
    extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    for camera_ex in extrinsics:
        camera_ex = camera_ex.as_matrix() @ rotX
        extrinsic = Transform(R.from_matrix(camera_ex[0:3, 0:3]), 
                                            camera_ex[0:3, 3])
    
        depth_img = compute_img(mesh, camera_ex ,camera_in)
        tsdf.integrate(depth_img, intrinsic, extrinsic)
        tsdf_high.integrate(depth_img, intrinsic, extrinsic)
    pc = tsdf_high.get_cloud()
    if pc.is_empty():
        print('Object rendering error, skip')
        return [], [], [], []
    
    # plan grasps
    state = State(tsdf, pc)
    grasps_ori, scores, _, qual_vol = grasp_plan_fn(state)
    if len(grasps_ori) == 0:
        print("No grasps found")
        return [], [], [], []

    p1_list = []
    p2_list = []
    data = []
    grasps = []
    sorted_indice = np.argsort(scores)
    for i in sorted_indice:
        grasps.append(grasps_ori[i])
    grasps.reverse()
    for i in range(min(len(grasps), 10)):
        p1 = grasps[i].pose * Transform(Rotation.identity(), [0.0, -grasps[i].width / 2, 0.05 / 2])
        p2 = grasps[i].pose * Transform(Rotation.identity(), [0.0, grasps[i].width / 2, 0.05 / 2])
        p1_list.append(p1.translation)
        p2_list.append(p2.translation)
    id1 = compute_all_id(p1_list, meshes)
    id2 = compute_all_id(p2_list, meshes)
    for j in range(len(p1_list)):
        tmp = (p1_list[j] - p2_list[j])/np.linalg.norm(p1_list[j] - p2_list[j])
        data_gg = np.hstack((np.array(id1[j]),np.array(p1_list[j]),-tmp,np.array(id2[j]),np.array(p2_list[j]),tmp))
        data.append(data_gg)
    print(f"Found {len(grasps)} grasps with scores: {scores.tolist()}")
    return grasps, scores, data, qual_vol

def compute_all_id(p, meshes):
    distance = np.zeros((len(meshes), len(p))) # (n, m)
    for id in range(len(meshes)):
        _, dis, _ = trimesh.proximity.closest_point(meshes[id], p)
        distance[id] = np.array(dis)
    return np.argmin(distance, axis=0)

class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
