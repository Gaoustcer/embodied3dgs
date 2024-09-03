import torch
import numpy as np
# from ffd import FFD_Multi_pose
# from peract.agent_pred_nextaction import CLIP_encoder
import torch.nn as nn
import clip
class CLIP_encoder(nn.Module):
    def __init__(self, device,model_name = "RN50"):
        super().__init__()
        self.device = device
        self.model, preprocess = clip.load(model_name, device=device, jit=False)
    
    @torch.no_grad()
    def encode_text(self, text):
        tokens = clip.tokenize(text)
        tokens = tokens.to(self.device)
        x = self.model.token_embedding(tokens).type(self.model.dtype)   # [B, T, D]

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)   # BTD -> TBD
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)   # TBD -> BTD
        x = self.model.ln_final(x).type(self.model.dtype)

        return x
    @torch.no_grad()
    def encode_onetoken(self, text):

        tokens = clip.tokenize(text).cuda()
        return self.model.encode_text(tokens)