import torch
import torch.nn as nn
import torch.nn.functional as F

from lbms_sam_base import GSEFE , MDFF, FeatureFusion




class LBMSSAM2Integration(nn.Module):
    def __init__(self, sam2_model, feature_dim = 256):
        super().__init__()
        self.sam2 = sam2_model

        self.fusion_conv = nn.Sequential([
            nn.Conv2d(feature_dim*4 , feature_dim, kernel_size = 3, padding_mode='same'),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        ])

     
    

    def forward(self, image_tensor):
        ##GSEFE
        gsefe_input = image_tensor
        gsefe_output = GSEFE(image = gsefe_input)
        
        
        ## MDFF
        encoder_output = self.SAM.modeling.backbones.image_encoder(image_tensor)
        hierarchical_features = encoder_output['backbone_fpn']
        mdff_output = MDFF(hierarchical_features)


    