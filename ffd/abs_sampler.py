from abc import ABC, abstractmethod
from ffd.point_cloud import Pointcloud
class AbstractPcdSampler(ABC):
    @abstractmethod
    def sample(pcd:Pointcloud,num_sample:int,*args,**kwargs) -> Pointcloud:
        pass
