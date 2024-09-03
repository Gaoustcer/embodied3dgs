# from ast import List
# from random import sample
from typing import List
from ffd.random_sampler import RandomSampler, OpacitySampler,SemanticCorrelation
from ffd.point_cloud import Pointcloud
from torch.utils.data import Dataset 
import os
import numpy as np
import torch
from torch.nn import functional as F

class FFD_Multi(Dataset):
    def __init__(self,ffd_rootpath,ae_rootpath,action_rootpath,sampler:RandomSampler,steps:List[str],num_sample:int,init_sample:int = None,sample_percent = 0.9,num_actions=4,*args,**kwargs) -> None:
        super(FFD_Multi,self).__init__()
        episodes = os.listdir(ffd_rootpath)
        self.proprio = []
        self.episodes = []
        self.actions = []
        self.sampler = sampler
        self.sample_size = num_sample
        if "semantic_feature" in kwargs.keys():
            self.semantic_feature = F.normalize(kwargs["semantic_feature"],dim = -1,p = 2)
        self.sample_percent = sample_percent
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
                # for feature_level in os.listdir(param_path):
                for feature_level in [""]:
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
                    epi_act = []
                    epi_state = []
                    self.num_actions = num_actions
                    for i in range(num_actions):
                        with open(action_path,"r") as fp:
                            action_file = json.load(fp)
                            if "endpose_grasp_{}".format(i) in action_file.keys():
                                action = action_file['endpose_grasp_{}'.format(i)]
                            else:
                                action = action_file['pose_{}'.format(i)]
                            # print('endpose_pregrasp_{}'.format(i))
                            # print(json.load(fp).keys())
                            # print(action_path)
                            if "endpose_gregrasp_{}".format(i) in action_file.keys():
                                pre_action = action_file['endpose_pregrasp_{}'.format(i)]
                            else:
                                pre_action = action_file['pose_{}'.format(i)]
                        def process(action):   
                            T = torch.tensor(action['pose'][:3]).to(torch.float32).cuda()
                            joint = torch.tensor(action['rotation']).to(torch.float32).cuda()
                            rotation = np.array(action['pose'][3:])
                            from scipy.spatial.transform import Rotation as R
                            quat = R.from_rotvec(rotation).as_quat()
                            quat = torch.from_numpy(quat).to(torch.float32).cuda()
                            gripper_open = torch.ones(1).to(torch.long).cuda()
                            # state = np.load(state_path,allow_pickle=True).item()
                            state = np.random.random(8)
                            # state = np.concatenate([state['quat'],state['T'],np.array([state['gripper open']])])
                            state = torch.from_numpy(state).to(torch.float32).cuda()
                            return [T,quat,gripper_open,joint],state
                        grasp_act,grasp_state = process(action)
                        pre_grasp_act,pre_grasp_state = process(pre_action)
                        epi_act.append([grasp_act,pre_grasp_act])
                        epi_state.append(grasp_state)

                    states.append(epi_state[0])
                    act.append(epi_act)
            if len(episode) != 0:
                self.actions.append(act)
                self.episodes.append(episode)
                self.proprio.append(states) # [8 dim tensor]
                self.args = args
                self.kwargs = kwargs
    
    def __len__(self):
        return len(self.episodes)
    def sample(self,index,actidx):
        return self.__getitem__(index,actidx=actidx)
    def __getitem__(self, index,actidx = None):
        episode = self.episodes[index]
        action = self.actions[index]
        proprio = self.proprio[index]
        from random import randint
        assert len(episode) == len(action)
        idx = randint(0,len(action) - 1)
        pcd = episode[idx]
        if actidx is None:
            act_idx = randint(0,self.num_actions - 1)
        else:
            act_idx = actidx
        # print("the pcd is",pcd)
        if isinstance(self.sampler,SemanticCorrelation):
            self.sampler: SemanticCorrelation
            # assert 
            sample_pcd = self.sampler.sample(pcd = pcd,num_sample = self.sample_size,semantic_query = self.semantic_feature[act_idx])
            pass
        else:
            sample_pcd = self.sampler.sample(pcd,self.sample_size,self.args,self.kwargs)
        return proprio[idx],sample_pcd.concat(),[action[idx][act_idx][0][0],action[idx][act_idx][0][1],action[idx][act_idx][0][2],action[idx][act_idx][0][3]],\
            [action[idx][act_idx][1][0],action[idx][act_idx][1][1],action[idx][act_idx][1][2],action[idx][act_idx][1][3]],act_idx # pcd,T,quat,grip
        # return super().__getitem__(index)
            
