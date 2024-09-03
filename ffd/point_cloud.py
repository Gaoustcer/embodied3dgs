import torch
import os
from autoencoder import Autoencoder
from autoencoder import ENCODER_HIDDEN_DIMS,DECODER_HIDDEN_DIMS
class Pointcloud:
    def __init__(self,
                 _xyz = torch.empty(1,3),
                 _rgb = torch.empty(1,3),
                 _scaling = torch.empty(1,3),
                 _rotation = torch.empty(1,4),
                 _opacity = torch.empty(1,1),
                 _lang_feature = torch.empty(1,512)):
        self._xyz = _xyz
        self._rgb = _rgb
        self._scaling = _scaling
        self._rotation = _rotation
        self._opacity = _opacity
        self._lang_feature = _lang_feature
        self._params = ["_rgb","_scaling","_rotation","_opacity","_lang_feature"]
    # @torch.no_grad()

    def load(self,ckpt_path,ae_path,sample_percent = 0.9,sample_size = None) -> None:
        # autoencoder = Autoencoder(ENCODER_HIDDEN_DIMS,DECODER_HIDDEN_DIMS).cuda()
        # autoencoder.load_state_dict(torch.load(ae_path))
        pcdparams = torch.load(ckpt_path,map_location="cuda")[0] 
        xyz = pcdparams[1]
        x_idx = (xyz[:,0] > -0.41) & (xyz[:,0] < 0.456)
        y_idx = (xyz[:,1] > 0.4) & (xyz[:,1] < 1.2)
        z_idx = (xyz[:,-1] > -0.35) & (xyz[:,-1] < 0.2)
        bounding_idx = x_idx & y_idx & z_idx
        # opac_quat = torch.quantile(pcdparams[6][idx].squeeze(),sample_percent)
        quat_idx = bounding_idx
        if sample_size != None:
            # print("quat",quat_idx.sum(),sample_size)
            idx = torch.randint(0,quat_idx.sum().item() - 1,(sample_size,)).long()
        else:
            idx = None
        # for i,t in pcdparams[1:]
        self._xyz = pcdparams[1][quat_idx][idx].squeeze()
        self._rgb = pcdparams[2][quat_idx][idx].squeeze()
        self._scaling = pcdparams[4][quat_idx][idx].squeeze()
        self._rotation = pcdparams[5][quat_idx][idx].squeeze()
        self._opacity = pcdparams[6][quat_idx][idx]
        with torch.no_grad():
            import torch.nn.functional as F
            if pcdparams[7][quat_idx][idx].squeeze().shape[0] == 3:
                pass
                # self._lang_feature = F.normalize(autoencoder.decode(pcdparams[7][quat_idx][idx].squeeze()),dim = -1,p = 2)
            else:
                self._lang_feature = F.normalize(pcdparams[7][quat_idx][idx].squeeze(), dim = -1, p = 2)
            
    def concat(self,position_randomize = None):
        if position_randomize is None:
            xyz = self._xyz
        else:
            xyz = torch.cat((self._xyz,torch.ones(self._xyz.size(0),1)),dim = -1)
            xyz = xyz @ position_randomize
            xyz = xyz[:,:3]
        return torch.cat([xyz] + [getattr(self,param) for param in self._params],dim = -1)

    def size(self):
        return self._xyz.shape[0]
    

    def sample(self,index:torch.Tensor):
        assert index.max() <= self._xyz.shape[0],"index should not be over-exceed than xyz shape"
        assert index.min() >= 0,"index should large than 0"
        return Pointcloud(
            _xyz = self._xyz[index],
            _rgb = self._rgb[index],
            _scaling = self._scaling[index],
            _rotation = self._rotation[index],
            _opacity = self._opacity[index],
            _lang_feature = self._lang_feature[index]
        )
        

if __name__ == "__main__":
    datasetpath = "../datasets/close_door/episode0/step_0/1_1/chkpnt30000.pth"
    weight = torch.load(datasetpath)
    for idx,param in enumerate(weight[0]):
        if isinstance(param,torch.Tensor):
            print(idx,param.shape)