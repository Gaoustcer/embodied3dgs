import open3d as o3d
import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pcd-path",type = str)
args = parser.parse_args()
pcd_path = args.pcd_path
dataset_path = os.path.dirname(pcd_path)
def create_camera_mesh(scale=1.0,color = [1,0,0]):
    """
    Creates a simple 3D camera mesh for visualization.

    Args:
    scale: float, the size of the camera mesh.

    Returns:
    open3d.geometry.TriangleMesh representing the camera.
    """
    camera = o3d.geometry.TriangleMesh()
    camera.vertices = o3d.utility.Vector3dVector(np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]) * scale)
    camera.triangles = o3d.utility.Vector3iVector(np.array([
        [1, 2, 3],
        [1, 3, 4],
        [2, 3, 4],
        [1, 2, 4]
    ]))
    camera.paint_uniform_color(color)
    return camera

def pose_to_transform(pose):
    x, y, z, Qx, Qy, Qz = pose
    
    # 平移向量
    t = np.array([x, y, z])
    
    # 旋转向量（轴角表示法）转换为旋转矩阵
    rotation_vector = np.array([Qx, Qy, Qz])
    # R.fro
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    # rotation_matrix = R.from_euler(seq = "xyz",angles = rotation_vector,degrees = False).as_matrix()
    
    # 构建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = t
    
    return transform

pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd])
# o3d.draw_geometries([pcd])
# posejson = os.path.join(dataset_path,"pose.json")
# from scipy.spatial.transform import Rotation as R
# with open(posejson,"r") as fp:
#     jsonfile = json.load(fp)
# cameras_poses = []
# pick_place_poses = []
# for key in jsonfile.keys():
#     if "scene" in key:
#         cameras_poses.append(jsonfile[key]['pose'])
#     else:
#         pick_place_poses.append(jsonfile[key]['pose'])
# # print("l)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# for camera_pose in cameras_poses:
#     print("create pose",camera_pose)
#     matrix = pose_to_transform(camera_pose)
#     camera = create_camera_mesh(color = [1,0,0])
#     camera.transform(matrix)
#     vis.add_geometry(camera)
# for pick_place_pose in pick_place_poses:
#     print("create pick and place pose",pick_place_pose)
#     matrix = pose_to_transform(pick_place_pose)
#     camera = create_camera_mesh(color = [0,1,0])
#     camera.transform(matrix)
#     vis.add_geometry(camera)
# vis.add_geometry(pcd)
# vis.run()
# vis.destroy_window()