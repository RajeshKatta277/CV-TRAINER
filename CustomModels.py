import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
import os
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F



class CropDiseaseModel(nn.Module):
    def __init__(self,num_crop_classes=None, num_disease_classes=None):
        super(CropDiseaseModel,self).__init__()
        self.backbone=models.resnet50(pretrained=True)
        in_features=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.crop_fc=nn.Linear(in_features, num_crop_classes)
        self.disease_fc=nn.Linear(2055, num_disease_classes)
        
    def forward(self, x):
        features=self.backbone(x)
        crop_output=self.crop_fc(features)
        combined=torch.cat((features,crop_output),dim=1)
        disease_output=self.disease_fc(combined)
        return crop_output, disease_output 
