import argparse
# from tkinter.tix import Tree
import open3d as o3d
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument("--pcd-root-path",type = str)
parser.add_argument("--down-sample-ratio",type = float)
args = parser.parse_args()
root_path: os.PathLike = args.pcd_root_path
ratio: float = args.down_sample_ratio

pcd_path = os.path.join(root_path,"point_cloud.pcd")
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)
N = points.shape[0]
sample_index = np.random.choice(N,int(N * ratio),replace=False)
points = points[sample_index]
colors = colors[sample_index]
if pcd.has_normals():
    normals = normals[sample_index]
else:
    normals = np.load(os.path.join(root_path,'features.npy'))
    normals = normals[sample_index]
    np.save(os.path.join(root_path,"features_{}.npy".format(ratio)),normals)
down_pcd = o3d.PointCloud()
down_pcd.points = o3d.utility.Vector3dVector(points)
down_pcd.colors = o3d.utility.Vector3dVector(colors)
if pcd.has_normals():
    down_pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.io.write_point_cloud(os.path.join(root_path,f"point_cloud_{ratio}.pcd"),down_pcd)