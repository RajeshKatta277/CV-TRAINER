from CFG import *
from Trainer import *
import torch
from utils import *
from Inference import *
from Engine import ENGINE

PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True

mode= 'training'


task='classification'  #segmentation, detection, classification, segformer 


train_param_dict=None

pred_param_dict=None

engine=ENGINE(task=task, mode=mode, train_param_dict=train_param_dict, pred_param_dict=pred_param_dict)

result= engine.run(image='/media/vassarml/HDD/AI_DATABASE/Crop_Disease_Photo_database/Maize/Maize_leaf spot/43.png') 




 

  





