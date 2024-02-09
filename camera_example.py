import os

import numpy as np
import trimesh
from matplotlib import pyplot as plt
import pyrender

from dataset.Dataset import Dataset

if __name__ == '__main__':
    # mesh = trimesh.load('/home/gdk/Repositories/DualArmManipulation/demo/banana.usdz')
    # mesh.show()

    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    dataset_dir = '/home/gdk/Repositories/DualArmManipulation/data/objects'
    object_id = '1191'
    dataset = Dataset(dataset_dir)
    meshes = dataset[object_id].load_meshes()
    mesh = trimesh.util.concatenate(meshes)

    # mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh = pyrender.Mesh.from_points(np.random.rand(100,3))

    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

    fx, fy, cx, cy = 540.0, 540.0, 320.0, 240.0
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)
    s = np.sqrt(2) / 2
    camera_pose = np.array([
        [0.0, -s, s, 2],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 2],
        [0.0, 0.0, 0.0, 1.0],
        ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle = np.pi / 16.0, outerConeAngle = np.pi / 6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()