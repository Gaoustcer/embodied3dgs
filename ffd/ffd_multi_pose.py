# from ast import List
# from random import sample
from typing import List
from ffd.random_sampler import RandomSampler, OpacitySampler,SemanticCorrelation,voxel_sampler
from ffd.point_cloud import Pointcloud
from torch.utils.data import Dataset 
import os
import numpy as np
import pdb
import torch
from torch.nn import functional as F
def sort_episode(dir_name):
    return int(dir_name.split("_")[1])
class FFD_Multi_pose(Dataset):
    def __init__(self,
            ffd_rootpath,
            ae_rootpath,
            action_rootpath,
            sampler:RandomSampler,
            steps:List[str],
            num_sample:int,
            init_sample:int = None,
            sample_percent = 0.9,
            num_tasks = 2,
            num_action_per_task = 4,
            feature_levels = [""],*args,**kwargs) -> None:
        super(FFD_Multi_pose,self).__init__()
        episodes = os.listdir(ffd_rootpath)
        episodes = sorted(episodes,key=sort_episode)
        self.episode_names = episodes
        self.proprio = []
        self.episodes = []
        self.actions = []
        self.sampler = sampler
        self.sample_size = num_sample
        if isinstance(self.sampler,voxel_sampler):
            assert "boundary" in kwargs.keys()
            self.boundary = kwargs['boundary']
            self.voxel_grid = kwargs['voxel_grid']
        if "semantic_feature" in kwargs.keys():
            self.semantic_feature = F.normalize(kwargs["semantic_feature"],dim = -1,p = 2)
        if "pose_randomize" in kwargs.keys():
            self.pose_randomize = kwargs['pose_randomize']
        else:
            self.pose_randomize = False
        if 'task_step' in kwargs.keys():
            self.task_step = kwargs["task_step"]
        else:
            self.task_step = None
        if "task_idx" in kwargs.keys():
            self.task_idx = kwargs['task_idx']
        else:
            self.task_idx = None
        self.sample_percent = sample_percent
        self.final_sampler = RandomSampler()
        from tqdm import tqdm
        print("episode",episodes)
        for index,ep in tqdm(enumerate(episodes)):
            # episode = []
            act = []
            states = []
            pcd_list = []
            for step in steps:
                ae_ckptpath = os.path.join(ae_rootpath,ep,step,"best_ckpt.pth")
                param_path = os.path.join(ffd_rootpath,ep,step)
                action_path = os.path.join(action_rootpath,ep,step,'pose.json')
                state_path = os.path.join(action_rootpath,ep,step,"state.npy")
                # for feature_level in os.listdir(param_path):
                # pcd_list = []
                for feature_level in feature_levels:
                    pcd = Pointcloud()
                    ckpt_path = os.path.join(param_path,feature_level,"chkpnt7000.pth")
                    if os.path.exists(ckpt_path) == False:
                        continue
                    else:
                        with torch.no_grad():
                            # for act_idx in range(len())
                            pcd.load(ckpt_path=os.path.join(param_path,feature_level,"chkpnt7000.pth"),ae_path=ae_ckptpath,sample_size = init_sample,sample_percent=self.sample_percent)
                            if isinstance(self.sampler,SemanticCorrelation):
                                for act_idx in range(self.semantic_feature.size(0)):
                                    pre_sample_pcd = self.sampler.sample(pcd = pcd,num_sample = self.sample_size,semantic_query = self.semantic_feature[act_idx])
                                    pcd_list.append(pre_sample_pcd)
                            elif isinstance(self.sampler,voxel_sampler):
                                pcd = self.sampler.sample(pcd,voxel_grid=self.voxel_grid,boundary=self.boundary[index]).squeeze()
                            if isinstance(self.sampler,SemanticCorrelation) == False:
                                pcd_list.append(pcd)
                            # if init_sample != None:
                                # pcd = self.sampler.sample(pcd,num_sample = init_sample,args = args,kwargs = kwargs)
                    # pcd = sampler.sample(pcd,num_sample)
                    # episode.append(pre_sample_pcd)
                    # action = np.load(action_path,allow_pickle=True).item()
                    import json
                    self.num_tasks = num_tasks
                    self.num_action_per_task = num_action_per_task
                    def process(action,current_gripper,past_gripper,past_end_ff):   
                        # print("past gripper is",past_gripper)
                        T = action['pose'][:3]
                        # T = torch.tensor(action['pose'][:3]).to(torch.float32)
                        joint = action['rotation']
                        # joint = torch.tensor(action['rotation']).to(torch.float32)
                        past_T = past_end_ff[:3]
                        # past_quat = past_end_ff[3:]
                        rotation = np.array(action['pose'][3:])
                        from scipy.spatial.transform import Rotation as R
                        past_quat = R.from_rotvec(past_end_ff[3:]).as_quat()
                        past_state = np.concatenate((past_T,past_quat,np.array([past_gripper])))
                        # past_state = torch.from_numpy(past_state).float()
                        quat = R.from_rotvec(rotation).as_quat()
                        # quat = torch.from_numpy(quat).to(torch.float32).float()
                        gripper_open = current_gripper
                        # gripper_open = torch.tensor(current_gripper).to(torch.long)
                        # state = np.load(state_path,allow_pickle=True).item()
                        # state = np.random.random(8)
                        # state = np.concatenate([state['quat'],state['T'],np.array([state['gripper open']])])
                        # state = torch.from_numpy(state).to(torch.float32)
                        # state 
                        return {"trans":T,"quats":quat,"gripper_state":gripper_open,"joint":joint,'state':past_state}
                    # self.num_actions = num_actions
                    # tasks = {
                    #     "task_1":None,
                    #     "task_0":None
                    # }
                    tasks = {}
                    for task_id in range(num_tasks):
                        tasks['task_{}'.format(task_id)] = None
                    for task in range(self.num_tasks):
                        task_action = {

                        }
                        # import pdb

                        # pdb.set_trace()
                        gripper_states = [1,0,0,1]
                        for task_step in range(self.num_action_per_task):
                            with open(action_path,"r") as fp:
                                action_file = json.load(fp)
                            act_idx = task * self.num_action_per_task + task_step
                            action = action_file['pose_{}'.format(act_idx)]
                            current_gripper_state = gripper_states[task_step]
                            if task_step == 0:
                                past_gripper_state = 1
                                past_end_ff_state = np.zeros(6)
                            else:
                                past_gripper_state = gripper_states[task_step - 1]
                                past_end_ff_state = action_file['pose_{}'.format(act_idx - 1)]['pose']
                            
                            
                            pre_process_action = process(action,current_gripper_state,past_gripper_state,past_end_ff_state)
                            for key in pre_process_action.keys():
                                if key not in task_action.keys():
                                    task_action[key] = [pre_process_action[key]]
                                else:
                                    task_action[key].append(pre_process_action[key])
                        tasks["task_{}".format(task)] = task_action
                            # action = action_file['endpose_grasp_{}'.format(i)]
                            # print('endpose_pregrasp_{}'.format(i))
                            # print(json.load(fp).keys())
                            # print(action_path)
                            # pre_action = action_file['endpose_pregrasp_{}'.format(i)]
                        
                        # grasp_act,grasp_state = process(action)
                        # pre_grasp_act,pre_grasp_state = process(pre_action)
                        # epi_act.append([grasp_act,pre_grasp_act])
                        # epi_state.append(grasp_state)

                    # states.append(epi_state[0])
                    # act.append(epi_act)
                    for key in tasks.keys():
                        for subkey in tasks[key].keys():
                            # pdb.set_trace()
                            tasks[key][subkey] = np.stack(tasks[key][subkey])
                            # print(tasks[key][subkey],key,subkey)
                            # tasks[key][subkey] = torch.stack(tasks[key][subkey])
                    act.append(tasks)
            if len(pcd_list) != 0:
                self.actions.append(act)
                self.episodes.append(pcd_list)
                # self.proprio.append(states) # [8 dim tensor]
                self.args = args
                self.kwargs = kwargs
    
    def __len__(self):
        return len(self.episodes)
    def sample(self,index,actidx):
        return self.__getitem__(index,actidx=actidx)
    @torch.no_grad()
    def process_action(self,action,matrix = None,actidx = None):
        # another_action
        # pdb.set_trace()
        if matrix == None:
            trans_actions = {}
            for key in action.keys():
                trans_actions[key] = action[key][actidx]
            return trans_actions
        else:
            # trans_action = {}
            trans = action['trans']
            quat = action['quats']
            from scipy.spatial.transform import Rotation as R
            Q_matrix = R.from_quat(quat).as_matrix()
            Pose_matrix = np.eye(4)
            Pose_matrix = np.repeat(Pose_matrix[None],Q_matrix.shape[0],axis = 0)
            Pose_matrix[:,:3,:3] = Q_matrix
            Pose_matrix[:,:3,3] = trans
            Pose_matrix = torch.from_numpy(Pose_matrix).float()
            New_matrix = torch.bmm(Pose_matrix,matrix.unsqueeze(0).repeat(Pose_matrix.size(0),1,1))
            new_trans = New_matrix[:,:3,3].cpu().numpy()
            new_rotation = New_matrix[:,:3,:3]
            new_rotation = new_rotation.cpu().numpy()
            new_quat = R.from_matrix(new_rotation).as_quat()
            trans_actions = {
                "trans": new_trans[actidx],
                "quats": new_quat[actidx]
            }
            states = np.zeros_like(action['state'])
            states[0] = action['state'][0]
            states[:,-1] = action['state'][:,-1]
            states[1:,:-1] = np.concatenate((new_trans[:-1],new_quat[:-1]),axis = -1)
            trans_actions['state'] = states[actidx]
            for key in ["gripper_state","joint"]:
                trans_actions[key] = action[key][actidx]
            return trans_actions
        
            # matrix = 
    def __getitem__(self, index,actidx = None):
        episode = self.episodes[index]
        action = self.actions[index] # list with one element
        import pdb
        # pdb.set_trace()
        # pdb.set_trace()
        # print("len action is",act)
        # proprio = self.proprio[index]
        from random import randint
        # assert len(episode) == len(action)
        idx = randint(0,len(action) - 1)
        
        if self.task_idx is not None:
            act_idx = self.task_idx
        elif actidx is None:
            act_idx = randint(0,self.num_tasks - 1)
        else:
            act_idx = actidx
        if self.task_step is not None:
            stage_idx = self.task_step
        else:
            stage_idx = randint(0,self.num_action_per_task - 1)
        if stage_idx > 1: # 2,3
            pcd = episode[2] # use the prompt Shelf for search
        else:
            pcd = episode[act_idx]
        # print("the pcd is",pcd)
        self.sampler: RandomSampler
        if isinstance(self.sampler,SemanticCorrelation):
            self.sampler: SemanticCorrelation
            # assert 
            sample_pcd = self.final_sampler.sample(pcd,self.sample_size,self.args,self.kwargs)
            # sample_pcd = self.sampler.sample(pcd = pcd,num_sample = self.sample_size,semantic_query = self.semantic_feature[act_idx])
            pass
        elif isinstance(self.sampler,voxel_sampler):
            self.sampler: voxel_sampler
            # pdb.set_trace()
            fts = pcd
        else:
            sample_pcd = self.sampler.sample(pcd,self.sample_size,self.args,self.kwargs)
        # import pdb
        if self.pose_randomize == False:
            matrix = None
        else:
            trans = np.random.random(3)
            rot_vec = np.random.random(3)
            from scipy.spatial.transform import Rotation as R
            R_matrix = R.from_rotvec(rot_vec).as_matrix()
            matrix = np.eye(4)
            matrix[:3,:3] = R_matrix
            matrix[:3,3] = trans
            matrix = torch.from_numpy(matrix).float()
        # pdb.set_trace()
        act_dict = self.process_action(action[idx]['task_{}'.format(act_idx)],matrix = matrix,actidx = stage_idx)
        np.concatenate
        
        if isinstance(self.sampler,voxel_sampler) == False:
            fts = sample_pcd.concat(position_randomize = matrix).permute(1,0)
        return {"fts": fts,
                "pc_centers": torch.zeros(3),
                "pc_radii": 1,
                'actions': np.concatenate((act_dict['trans'],act_dict['quats'],np.array([act_dict['gripper_state']])),axis = -1),
                "prev_actions": act_dict['state'],
                "step_ids": stage_idx,
                "taskvar_ids":act_idx,
                'txt_masks': torch.ones(77) == 1,
                "epi_name": index
                } # without epiosode_ids/taskvars(str) and instr_embeds
        return  sample_pcd.concat(position_randomize = matrix),self.process_action(action[idx]['task_{}'.format(act_idx)],matrix = matrix,actidx = stage_idx),act_idx,stage_idx
        # return 
        # return proprio[idx],sample_pcd.concat(),[action[idx][act_idx][0][0],action[idx][act_idx][0][1],action[idx][act_idx][0][2],action[idx][act_idx][0][3]],\
        #     [action[idx][act_idx][1][0],action[idx][act_idx][1][1],action[idx][act_idx][1][2],action[idx][act_idx][1][3]],act_idx # pcd,T,quat,grip
        # return super().__getitem__(index)
            
