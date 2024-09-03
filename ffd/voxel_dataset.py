import torch
from torch.utils.data import Dataset 
import os
class Voxel_dataset(Dataset):
    def __init__(self,task,root_path="datasets/voxel"):
        taskrootpath = os.path.join(root_path,task)
        self.data = {} 
        for taskfile in os.listdir(taskrootpath):
            taskpath = os.path.join(taskrootpath,taskfile)
            ckpt = torch.load(taskpath)
            for key in ckpt.keys():
                self.data.setdefault(key,[])
                self.data[key].append(ckpt[key])
        for key in self.data.keys():
            self.data[key] = torch.cat(self.data[key],dim = 0)
        self.data['voxelize_representation'] = self.data['voxelize_representation'].permute(0,4,1,2,3)

    def __len__(self):
        return self.data['gt_actions'].size(0)
    def __getitem__(self,idx):
        return {key:self.data[key][idx] for key in self.data.keys()}
            #    self.data.