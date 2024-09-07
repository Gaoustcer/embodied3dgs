from re import T
from typing import List
import torch
import os
import numpy as np
import argparse
import clip
from torch.nn import functional as F
# from autoencoder import Autoencoder
# from autoencoder import ENCODER_HIDDEN_DIMS,DECODER_HIDDEN_DIMS
from matplotlib import pyplot as plt
from utils.sh_utils import SH2RGB
parser = argparse.ArgumentParser()

parser.add_argument("--object",type = str)
parser.add_argument("--obj-name",type=str)
parser.add_argument("--feature-path",type = str)
parser.add_argument("--top-k",type = int,default=10240)
# parser.add_argument("--ckpt-path",type = str)

args = parser.parse_args()
# object:List[str] = [args.object]
print("feature path",args.feature_path,args.object,args.obj_name)
features = torch.load(args.feature_path,map_location="cuda")[0]
xyz = features[1].cpu().detach().numpy()
colors = features[2].cpu().detach().squeeze().numpy()
colors = SH2RGB(colors)
names = ["x","y","z"]

x_idx = (xyz[:,0] > -0.41) & (xyz[:,0] < 0.456)
y_idx = (xyz[:,1] > 0.4) & (xyz[:,1] < 1.2)
z_idx = (xyz[:,-1] > -0.35) & (xyz[:,-1] < 0.2)
idx = x_idx & z_idx & y_idx
# idx = x_idx & z_idx
# idx = z_idx
# idx = x_idx & y_idx
opacity = features[-7].squeeze()
features = features[-6]

# print(opacity.shape)

# topkidx = torch.topk(opacity,1024 * 32).indices
print("origin xyz",xyz.shape)
features = features[idx]
xyz = xyz[idx]
print("bounding xyz",xyz.shape)
# exit()
for id,name in enumerate(names):
    array = xyz[:,id]
    print(array.shape,xyz.shape)
    plt.hist(array,bins=1024)
    plt.savefig(os.path.join(os.path.dirname(args.feature_path),f"{args.obj_name}_dis_{name}.png"))
    plt.cla()
sample_idx = torch.randint(0,xyz.shape[0] - 1,(1227680//3 ,))
features = features[sample_idx]
xyz = xyz[sample_idx]
model,_ = clip.load("ViT-B/16",device = "cuda")
tokenize = clip.tokenize([args.object]).cuda()
os.makedirs(os.path.join(os.path.dirname(args.feature_path),"3d_query"),exist_ok=True)
with torch.no_grad():
    text_embedding:torch.Tensor = model.encode_text(tokenize)
    text_embedding = text_embedding/torch.norm(text_embedding, p = 2)
    text_embedding = text_embedding.float()
    # ae = Autoencoder(ENCODER_HIDDEN_DIMS,DECODER_HIDDEN_DIMS).cuda()
    # ae.load_state_dict(torch.load(args.ckpt_path))
    # if features.size(-1) != 512:
        # features = ae.decode(features)
    features = F.normalize(features, p = 2,dim = -1)
    scores = text_embedding @ features.T
    print("test_embedding shape",text_embedding.shape)
    print("feature shape",features.shape)
    scores = scores.squeeze()
    indices = torch.topk(scores,args.top_k).indices
    
    # scores = scores[indices]
    # features = features[indices]
    indices = indices.detach().cpu().numpy()
    # xyz = xyz[indices]

    scores = scores.squeeze().detach().cpu().numpy() * 2
    
     
    print("score shape",scores.shape,scores.max(),scores.min(),scores.mean())
    plt.hist(scores,bins = 32)
    plt.savefig(os.path.join(os.path.dirname(args.feature_path),"3d_query/feature_{}_score.png".format(args.obj_name)))
    plt.cla()
    cmap = plt.get_cmap("BuGn")
    # colors = colors[idx]
    # print("color is",colors.shape)
    colors = cmap(scores)[:,:3]
    colors[indices] = np.array([1,0,0])
    print(colors.shape)
    # xyz = xyz
    import open3d as o3d
    pcd = o3d.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plt.scatter(xyz[:,0],xyz[:,1],s = 0.1)
    import numpy as np
    print("mean xyz for object {}".format(args.object),np.mean(xyz,axis = 0))
    plt.savefig(os.path.join(os.path.dirname(args.feature_path),"3d_query/xy_{}_distribution.png".format(args.obj_name)))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(args.feature_path),"3d_query/feature_{}_bounding_topk.pcd".format(args.obj_name)),pcd)

