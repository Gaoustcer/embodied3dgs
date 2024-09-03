from ffd.abs_sampler import AbstractPcdSampler
from ffd.point_cloud import Pointcloud
import torch
from voxel_grid import VoxelGrid

class RandomSampler(AbstractPcdSampler):
    def sample(self,pcd: Pointcloud, num_sample: int, *args, **kwargs) -> Pointcloud:
        # return super().sample(num_sample, *args, **kwargs)
        pcd_size = pcd.size()
        index = torch.randint(0,pcd_size,(num_sample,))
        return pcd.sample(index)

class OpacitySampler(AbstractPcdSampler):
    def sample(self,pcd: Pointcloud, num_sample: int, *args, **kwargs) -> Pointcloud:
        # print("sample pcd",pcd)
        opacity = pcd._opacity
        _,index = torch.sort(opacity,descending=True)
        index = index[:num_sample]
        return pcd.sample(index)
    
class voxel_sampler(AbstractPcdSampler):
    def sample(self,pcd:Pointcloud,voxel_grid:VoxelGrid,boundary:torch.Tensor):
        if boundary.dim() == 1:
            boundary = boundary.unsqueeze(0)
        return voxel_grid.coords_to_bounding_voxel_grid(pcd._xyz.unsqueeze(0),pcd._lang_feature.unsqueeze(0),coord_bounds=boundary)
    pass
class BoundingSampler(AbstractPcdSampler):
    def sample(self,pcd: Pointcloud, num_sample: int, *args, **kwargs) -> Pointcloud:
        # return super().sample(num_sample, *args, **kwargs)
        xyz = pcd._xyz
        x_idx = (xyz[:,0] > -0.41) & (xyz[:,0] < 0.456)
        y_idx = (xyz[:,1] > 0.4) & (xyz[:,1] < 1.2)
        z_idx = (xyz[:,-1] > -0.35) & (xyz[:,-1] < 0.2)
        idx = x_idx & z_idx & y_idx
        idx = torch.nonzero(idx).squeeze()
        # pcd_size = pcd.size()
        ind = torch.randint(0,idx.shape[0] - 1,(num_sample,))
        index = idx[ind]
        # index = torch.randint(0,pcd_size,(num_sample,)).cuda()
        return pcd.sample(index)
        # return super().sample(num_sample, *args, **kwargs)

class SemanticCorrelation(AbstractPcdSampler):
    @torch.no_grad()
    def sample(self,pcd:Pointcloud, num_sample: int,semantic_query:torch.Tensor,*args,**kwargs) -> Pointcloud:
        import torch.nn.functional as F
        # semantic_query = semantic_query.cpu()
        xyz = pcd._xyz
        x_idx = (xyz[:,0] > -0.41) & (xyz[:,0] < 0.456)
        y_idx = (xyz[:,1] > 0.4) & (xyz[:,1] < 1.2)
        z_idx = (xyz[:,-1] > -0.35) & (xyz[:,-1] < 0.2)
        idx = x_idx & z_idx & y_idx
        idx = torch.nonzero(idx).squeeze()
        boundingpcd = pcd.sample(idx)
        
        score = semantic_query.float() @ F.normalize(boundingpcd._lang_feature,dim = -1,p = 2).T
        # print("score shape",score.shape)
        max_score_index = torch.topk(score,2 * num_sample).indices
        return boundingpcd.sample(max_score_index)
        index = max_score_index[torch.randint(0,max_score_index.shape[0] - 1,(num_sample,))]
        return boundingpcd.sample(index)
        pass