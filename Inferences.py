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



import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection import FCOS  
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch import nn
import torchvision.models.resnet as R
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
import numpy as np


import torch
import rasterio
import os
import geopandas as gpd
from rasterio.features import geometry_mask
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchgeo.samplers import GridGeoSampler
import numpy as np   
import pytorch_lightning as pl
from lightning.pytorch import Trainer
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchgeo.datasets import RasterDataset
from torchgeo.datasets import BoundingBox
from torchgeo.datasets import GeoDataset
from torchvision.transforms.functional import to_tensor
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from shapely.geometry import box
from rasterio.features import rasterize
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch import Unet, Linknet, PSPNet
from torchgeo.models import FCN
from torchgeo.datasets import RasterDataset, VectorDataset, IntersectionDataset, GeoDataset
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import (
    stack_samples,
    RasterDataset,
    unbind_samples,
)

from torchgeo.datamodules import GeoDataModule
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datasets.splits import random_bbox_assignment 
from torchgeo.samplers import RandomGeoSampler
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import MetricCollection

from torchgeo.samplers import (
    RandomGeoSampler, 
    RandomBatchGeoSampler, 
    GridGeoSampler
) 

from rasterio.plot import show
from rasterio.merge import merge
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.crs import CRS
from rasterio.io import MemoryFile
import timeit
from CustomModels import *
 

 
        
             
class Inference_segmentation:
    def __init__(self,model=None,accelerator='cpu',checkpoint=None,backbone='resnet50',in_channels=3,num_classes=2):
        self.model_name=model
        self.accelerator=accelerator
        self.checkpoint_path=checkpoint
        self.backbone=backbone
        self.in_channels=in_channels
        self.classes=num_classes
        self.model_map = {
		    'unet': smp.Unet,
		    'unet++': smp.UnetPlusPlus,
		    'deeplabv3+': smp.DeepLabV3Plus,
		    'deeplabv3': smp.DeepLabV3,
		    'linknet': smp.Linknet,
		    'fcn': FCN, }
        self.model=self.load_model()
    def load_model(self):
        if self.model_name not in self.model_map:
            raise ValueError(f"Model {self.model_name} is not supported. Supported models: {list(self.model_map.keys())}")
        
        model_class = self.model_map[self.model_name]
        model = model_class(
            encoder_name=self.backbone, 
            in_channels=self.in_channels, 
            classes=self.classes
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cuda'))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v
        model_state_dict=model.state_dict()
        f_state_dict={k: v for k,v in new_state_dict.items() if k in model_state_dict}
        model.load_state_dict(f_state_dict)
        model.eval()
        return model
        

    def create_in_memory_geochip(self, predicted_chip, geotransform, crs):
        photometric = 'MINISBLACK'
        memfile = MemoryFile()
        dataset = memfile.open(
            driver='GTiff',
            height=predicted_chip.shape[1],
            width=predicted_chip.shape[2],
            count=predicted_chip.shape[0], 
            dtype=np.uint8,
            crs=crs,
            transform=geotransform,
            photometric=photometric,
        )
        dataset.write(predicted_chip)
        return dataset
      
    def georreferenced_chip_generator(self, dataloader, crs, pixel_size):
        georref_chips_list = []
        for i, sample in enumerate(dataloader):
            image, gt_mask, bbox = sample['image'], sample['mask'], sample['bbox'][0]
            image = image / 255.0  # as I'm not using a GeoDatamodule, I need to divide the images by 255 manually 
            prediction = self.model.predict(image)
            prediction = torch.softmax(prediction, dim=1)
            prediction = torch.argmax(prediction, dim=1) 
            geotransform = from_origin(bbox.minx, bbox.maxy, pixel_size, pixel_size)
            georref_chips_list.append(self.create_in_memory_geochip(prediction, geotransform, crs)) 
        return georref_chips_list 
      
    def merge_georeferenced_chips(self, chips_list, output_path):
        # Merge the chips using Rasterio's merge function
        merged, merged_transform = merge(chips_list)

        # Calculate the number of rows and columns for the merged output
        rows, cols = merged.shape[1], merged.shape[2]
    
        merged_metadata = chips_list[0].meta
        merged_metadata.update({
            'height': rows,
            'width': cols,
            'transform': merged_transform
        }) 
    
        with rasterio.open(output_path, 'w', **merged_metadata) as dst:
            dst.write(merged)
          
        for chip in chips_list:
            chip.close()
          
    def predict(self, image, output_dir, tile_size):
        self.test_dataset = RasterDataset(paths=image)
        self.test_sampler = GridGeoSampler(self.test_dataset, size=tile_size, stride=tile_size)
        self.test_loader = DataLoader(self.test_dataset, sampler=self.test_sampler, collate_fn=stack_samples)
        self.pixel_size = self.test_dataset.res
        self.crs = self.test_dataset.crs.to_epsg()
        start = timeit.default_timer() 
        chips_generator = self.georreferenced_chip_generator(self.test_loader, self.crs, self.pixel_size)
        print("The time taken to predict was: ", timeit.default_timer() - start)
        start = timeit.default_timer()  # Measuring the time
        file_name = os.path.join(output_dir, os.path.basename(image))
        self.merge_georeferenced_chips(chips_generator, file_name)
        print("The time taken to generate a georeferenced image and save it was: ", timeit.default_timer() - start) 
        
        
        


class InferenceModelLoader:
    BACKBONE_LAT_DIM_MAP = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        'resnext50_32x4d': 2048,
        'resnext101_32x8d': 2048,
        'wide_resnet50_2': 2048,
        'wide_resnet101_2': 2048,
    }

    BACKBONE_WEIGHT_MAP = {
        'resnet18': R.ResNet18_Weights.DEFAULT,
        'resnet34': R.ResNet34_Weights.DEFAULT,
        'resnet50': R.ResNet50_Weights.DEFAULT,
        'resnet101': R.ResNet101_Weights.DEFAULT,
        'resnet152': R.ResNet152_Weights.DEFAULT,
        'resnext50_32x4d': R.ResNeXt50_32X4D_Weights.DEFAULT,
        'resnext101_32x8d': R.ResNeXt101_32X8D_Weights.DEFAULT,
        'wide_resnet50_2': R.Wide_ResNet50_2_Weights.DEFAULT,
        'wide_resnet101_2': R.Wide_ResNet101_2_Weights.DEFAULT,
    }

    def __init__(self, model, backbone, checkpoint_path, accelerator,in_channels, num_classes,labels_map=None):
        self.model_name= model
        self.backbone_name = backbone
        self.checkpoint_path = checkpoint_path
        self.device = accelerator
        self.num_classes = num_classes
        self.in_channels=in_channels
        self.labels_map=labels_map
        self.model = self.load_model()

    def load_model(self):
        if self.backbone_name not in self.BACKBONE_LAT_DIM_MAP:
            raise ValueError(f"Backbone type '{self.backbone_name}' is not valid.")
        kwargs={'backbone_name':self.backbone_name,
               }
        kwargs['weights']=None
        
        
        latent_dim = self.BACKBONE_LAT_DIM_MAP[self.backbone_name]

        if self.model_name == 'faster-rcnn':
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), 
                aspect_ratios=((0.5, 1.0, 2.0))
            )
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
            )
            model = FasterRCNN(
                model_backbone,
                self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif self.model_name == 'fcos':
            kwargs['extra_blocks']=feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs['norm_layer']=(misc.FrozenBatchNorm2d if weights else torch.nn.BatchNorm2d)
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,))
            )
            model = FCOS(
                model_backbone, self.num_classes, anchor_generator=anchor_generator
            )
        elif self.model_name == 'retinanet':
            kwargs['extra_blocks']=feature_pyramid_network.LastLevelP6P7(latent_dim, 256)
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_sizes = (
                (16, 20, 25),
                (32, 40, 50),
                (64, 80, 101),
                (128, 161, 203),
                (256, 322, 406),
                (512, 645, 812),
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            head = RetinaNetHead(
                model_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                self.num_classes,
                norm_layer=partial(nn.GroupNorm, 32),
            )
            model = RetinaNet(
                model_backbone,
                self.num_classes,
                anchor_generator=anchor_generator,
                head=head,
            )
        else:
            raise ValueError(f"Model type '{self.model_name}' is not valid.")
            
            
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v
        model_state_dict=model.state_dict()
        f_state_dict={k: v for k,v in new_state_dict.items() if k in model_state_dict}
        model.load_state_dict(f_state_dict)
        model=model.to(torch.device(self.device))
        model.eval()
        
        print('model loaded!')
        return model
        
        
        
    def forward(self,image=None):
        image=Image.open(image).convert('RGB')
        transform=T.Compose([T.ToTensor()])
        image=transform(image).unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            prediction=self.model(image)
        return image.squeeze(0).permute(1,2,0).cpu(), prediction[0] 
        
        
    def predict(self,image,nms_threshold=0.1,output_dir=None, save_result=False,min_confidence=0.10):
     
        img,preds=self.forward(image)
        indices = nms(preds['boxes'], preds['scores'], nms_threshold)
        image_np = img*0.5 + 0.5 
        fig, ax = plt.subplots(1, figsize=(6,4))
        ax.imshow(image_np)
        lab=[]
        scr=[]
        for idx in indices:
          x_min, y_min, x_max, y_max = preds['boxes'][idx].cpu().numpy()
          labels, scores=preds['labels'][idx].cpu().numpy(), preds['scores'][idx].cpu().numpy()
          if scores<min_confidence:
              continue
          print(self.labels_map[int(labels)], scores)
          
          lab.append(self.labels_map[int(labels)])
          scr.append(scores)
          rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
          ax.add_patch(rect)
          plt.text(x_min,y_min,s=f'{self.labels_map[int(labels)]} confidence:{scores*100:.2f}%',bbox=dict(facecolor='red', alpha=0.5))
        if save_result:
            plt.savefig(os.path.join(output_dir,os.path.basename(image)) )
            
        plt.show()
        
        return {'image': img, 'labels': lab, 'scores': scr} 
    
        
                
        
 



class ClassificationInference:
    def __init__(self,model=None,checkpoint_path=None,in_channels=3, num_classes=None,image_size=224, accelearator='cuda',labels_map=None):
        self.model_name=model
        self.checkpoint_path=checkpoint_path
        self.in_channels=in_channels
        self.image_size=image_size
        self.num_classes=num_classes
        self.device=accelearator
        self.model=self.load_model()
        self.labels_map=labels_map
        self.transform=transforms.Compose([
                  transforms.Resize((self.image_size,self.image_size)),
                  transforms.ToTensor(),
                  ])
    def load_model(self):
        model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            in_chans=self.in_channels,
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cuda'))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict'] 
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v
        model_state_dict=model.state_dict()
        f_state_dict={k: v for k,v in new_state_dict.items() if k in model_state_dict}
        model.load_state_dict(f_state_dict)
        print(model)
        print('model loaded!')
        return model
    def predict(self,image=None):
        image=Image.open(image).convert('RGB')
        if self.transform:
            image=self.transform(image)
        image=image.unsqueeze(0)
        with torch.no_grad():
            pred=self.model1(image)
        pred=F.softmax(pred,dim=1)
        cls=torch.argmax(pred,dim=1)
        
        return {'image':image.squeeze(0).permute(1,2,0),'label': self.labels_map[int(cls)]}
        
        
        
class CropDiseaseClassificationInference:
    def __init__(self,model=None,checkpoint_path=None, in_channels=3, num_crop_classes=None ,num_disease_classes=None,image_size=224, device='cuda'):
        self.model_name=model
        self.checkpoint1=checkpoint_path
        self.in_channels=in_channels
        self.image_size=image_size
        self.num_crop_classes=num_crop_classes
        self.num_disease_classes=num_disease_classes
        self.device=device
        self.model1=self.load_model(self.checkpoint1)
        self.transform=transforms.Compose([
                  transforms.Resize((self.image_size,self.image_size)),
                  transforms.ToTensor(),
                  ])
        self.labels_map={'crop': {0: 'Groundnut', 1: 'Sugarcane', 2: 'Paddy', 3: 'Wheat', 4: 'Cotton', 5: 'Maize', 6: 'Mustard'}, 'disease': {0: 'early_rust', 1: 'healthy leaf', 2: 'nutrition deficiency', 3: 'early_leaf_spot', 4: 'late leaf spot', 5: 'rust', 6: 'RedRust', 7: 'Sugarcane__rust', 46: 'Bacterial Blight', 9: 'Sugarcane__red_rot', 10: 'Sugarcane__red_stripe', 11: 'Sugarcane__healthy', 12: 'Red Rot', 13: 'Yellow', 14: 'Sugarcane__bacterial_blight', 15: 'Healthy', 16: 'Rice__neck_blast', 17: 'Rice_tungro', 18: 'Rice_downy_mildew', 19: 'Rice_bacterial_panicle_blight', 20: 'Rice_hispa', 21: 'Rice__leaf_blast', 22: 'Rice_bacterial_leaf_streak', 23: 'Rice_blast', 24: 'Rice_dead_heart', 25: 'Rice_brown_spot', 26: 'Rice_healthy', 27: 'Rice_bacterial_leaf_blight', 28: 'Smut', 29: 'Leaf Blight', 30: 'Mildew', 31: 'Common Root Rot', 32: 'Stripe rust', 33: 'Black Rust', 34: 'Aphid', 35: 'Mite', 36: 'Wheat__yellow_rust', 37: 'Tan spot', 38: 'Stem fly', 39: 'Blast', 40: 'Wheat__healthy', 41: 'Wheat__septoria', 42: 'Wheat__brown_rust', 43: 'Fusarium Head Blight', 44: 'Target spot', 45: 'Healthy leaf', 47: 'Powdery Mildew', 48: 'Army worm', 49: 'curl_virus', 50: 'Aphids', 51: 'fussarium_wilt', 52: 'Corn__gray_leaf_spot', 53: 'Maize_leaf beetle', 54: 'Corn__healthy', 55: 'Maize Gray_Leaf_Spot', 56: 'Maize_grasshoper', 57: 'Maize Common_Rust', 58: 'Maize_fall armyworm', 59: 'Maize_healthy', 60: 'Maize_streak virus', 61: 'Corn__northern_leaf_blight', 62: 'Corn__common_rust', 63: 'Maize_leaf spot', 64: 'Maize_leaf blight', 65: 'SIAP PANEN', 66: 'BELUM SIAP'}} 

        
    def load_model(self,checkpoint_path=None):
        model = CropDiseaseModel(num_crop_classes=self.num_crop_classes, num_disease_classes=self.num_disease_classes)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v
        model_state_dict=model.state_dict()
        f_state_dict={k: v for k,v in new_state_dict.items() if k in model_state_dict}
        model.load_state_dict(f_state_dict)
        print('model loaded!')
        return model
        
    def predict(self,image=None):
        image_path=image
        image=Image.open(image).convert('RGB')
        if self.transform:
            image=self.transform(image)
        image=image.unsqueeze(0)
        with torch.no_grad():
            crop1, disease1=self.model1(image)

        crop1,disease1=F.softmax(crop1,dim=1),F.softmax(disease1,dim=1)
        crop=torch.argmax(crop1,dim=1)
        disease=torch.argmax(disease1,dim=1) 
        
        plt.figure(figsize=(8, 6))
    
    
        plt.imshow(image.squeeze(0).permute(1,2,0), cmap='gray' if image.squeeze(0).ndim == 2 else None)
    
    # Add the text to the image
        plt.text(10, 10, f" CROP:  {self.labels_map['crop'][int(crop)]},   DISEASE: {self.labels_map['disease'][int(disease)]}", color='blue', fontsize=12, weight='bold', ha='left', va='top')
    
    # Hide axes
        plt.axis('off')
        
        plt.savefig(os.path.join('/media/vassarml/HDD/Rajesh/test_seg/outputs/crop_disease',os.path.basename(image_path)) )
    
    # Show the image
        plt.show()
        
        return {'image':image.squeeze(0).permute(1,2,0), 'crop':self.labels_map['crop'][int(crop)], 'disease':self.labels_map['disease'][int(disease)]} 
        
        
            
            


