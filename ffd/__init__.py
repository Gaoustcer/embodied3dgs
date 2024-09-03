# from ast import List
# from random import sample
from typing import List
from ffd.random_sampler import RandomSampler, OpacitySampler,BoundingSampler,SemanticCorrelation,voxel_sampler
from ffd.point_cloud import Pointcloud
from torch.utils.data import Dataset 
import os
import numpy as np
import torch
from .ffd_multi import FFD_Multi
from .ffd_multi_pose import FFD_Multi_pose
import torch.nn.functional as F
from .Lang_encoder import CLIP_encoder
from .voxel_dataset import Voxel_dataset
class FFD(Dataset):
    def __init__(self,ffd_rootpath,ae_rootpath,action_rootpath,sampler:RandomSampler,steps:List[str],num_sample:int,init_sample:int = None,sample_percent = 0.9,act_idx = 0,*args,**kwargs) -> None:
        super(FFD,self).__init__()
        episodes = os.listdir(ffd_rootpath)
        self.proprio = []
        self.episodes = []
        self.actions = []
        self.sampler = sampler
        self.sample_size = num_sample
        self.sample_percent = sample_percent
        if "semantic_embeddings" in kwargs.keys():
            self.semantic_embeddings = F.normalize(kwargs['semantic_embeddings'],p = 2,dim = -1)
        from tqdm import tqdm
        for ep in tqdm(episodes):
            episode = []
            act = []
            states = []
            for step in steps:
                ae_ckptpath = os.path.join(ae_rootpath,ep,step,"best_ckpt.pth")
                param_path = os.path.join(ffd_rootpath,ep,step)
                action_path = os.path.join(action_rootpath,ep,step,'pose.json')
                state_path = os.path.join(action_rootpath,ep,step,"state.npy")
                for feature_level in os.listdir(param_path):
                    pcd = Pointcloud()
                    ckpt_path = os.path.join(param_path,feature_level,"chkpnt7000.pth")
                    if os.path.exists(ckpt_path) == False:
                        continue
                    else:
                        with torch.no_grad():
                            pcd.load(ckpt_path=os.path.join(param_path,feature_level,"chkpnt7000.pth"),ae_path=ae_ckptpath,sample_size = init_sample,sample_percent=self.sample_percent)
                            # if init_sample != None:
                                # pcd = self.sampler.sample(pcd,num_sample = init_sample,args = args,kwargs = kwargs)
                    # pcd = sampler.sample(pcd,num_sample)
                    episode.append(pcd)
                    # action = np.load(action_path,allow_pickle=True).item()
                    import json
                    with open(action_path,"r") as fp:
                        action_data = json.load(fp)
                        if "endpose_{}".format(act_idx) in action_data.keys():
                            action = action_data['endpose_{}'.format(act_idx)]
                        else:
                            action = action_data['endpose_grasp_{}'.format(act_idx)]
                    T = torch.tensor(action['pose'][:3]).to(torch.float32)
                    joint = torch.tensor(action['rotation']).to(torch.float32)
                    rotation = np.array(action['pose'][3:])
                    from scipy.spatial.transform import Rotation as R
                    quat = R.from_rotvec(rotation).as_quat()
                    quat = torch.from_numpy(quat).to(torch.float32)
                    gripper_open = torch.ones(1).to(torch.long)
                    # state = np.load(state_path,allow_pickle=True).item()
                    state = np.random.random(8)
                    # state = np.concatenate([state['quat'],state['T'],np.array([state['gripper open']])])
                    state = torch.from_numpy(state).to(torch.float32)
                    states.append(state)
                    act.append([T,quat,gripper_open,joint])
            if len(episode) != 0:
                self.actions.append(act)
                self.episodes.append(episode)
                self.proprio.append(states) # [8 dim tensor]
                self.args = args
                self.kwargs = kwargs
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, index):
        episode = self.episodes[index]
        action = self.actions[index]
        proprio = self.proprio[index]
        from random import randint
        assert len(episode) == len(action)
        idx = randint(0,len(action) - 1)
        pcd = episode[idx]
        # print("the pcd is",pcd)
        
        sample_pcd = self.sampler.sample(pcd,self.sample_size,self.args,self.kwargs)
        
        return proprio[idx],sample_pcd.concat(),action[idx][0],action[idx][1],action[idx][2],action[idx][3] # pcd,T,quat,grip
        # return super().__getitem__(index)
            
