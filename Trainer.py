
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
from LitModules import *
from DataModules import * 
from Inference_Segmentation import *
#from Inference_Detection import *
from utils import *
from CFG import *

 


class Custom_Trainer:
    def __init__(self, task=None,cfg=None):
        self.cfg=cfg
        self.finetune=self.cfg.finetune
        self.checkpoint=self.cfg.checkpoint
        
 
          
        self.callbacks=[
		         ModelCheckpoint(dirpath=self.cfg.checkpoint_dir,filename=self.cfg.checkpoint_name,save_top_k=self.cfg.top_k_models,mode=self.cfg.monitor_mode,monitor=self.cfg.monitor,save_last=True,verbose=True)
		]

        if self.cfg.earlystopping:
            self.callbacks.append(EarlyStopping(monitor=self.cfg.monitor,mode=self.cfg.monitor_mode ,patience=self.cfg.patience)) 
        self.logger=None
        self.trainer = Trainer(
            accelerator=self.cfg.accelerator,
            devices=self.cfg.devices,
            min_epochs=self.cfg.min_epochs,
            max_epochs=self.cfg.max_epochs,
            callbacks=self.callbacks,
            logger=self.logger,
        ) 
    
            
 
            
        

    def initialize_segmentation(self):
       
      
        if self.cfg.model=='segformer':
            self.datamodule=SegFormerDataModule(
               image_dir=self.cfg.image_dir,
	        mask_dir=self.cfg.mask_dir,
	        batch_size=self.cfg.batch_size, 
	        patch_size=self.cfg.patch_size,  
	        length=self.cfg.length,  
	        num_workers=self.cfg.num_workers,  
	        collate_fn=stack_samples,  
	        train_split=self.cfg.train_split
	         )
            if self.finetune:
                self.task=SegFormerLitModule.load_from_checkpoint(self.checkpoint)
                print(f'weights of {self.checkpoint} loaded successfully')
            else:
	         
                self.task=SegFormerLitModule(
			model=self.cfg.model,
			num_classes=self.cfg.num_classes,
			ignore_index=self.cfg.ignore_index,
			lr=self.cfg.lr,
			patience=self.cfg.patience, 
			optimizer=self.cfg.optimizer
			)
            
        else:
            if self.cfg.Kfold==True:
                if self.cfg.tiles:
                    self.datamodule=KfoldTileSegmentationDataModule(
                       image_dir=self.cfg.image_dir,
			mask_dir=self.cfg.mask_dir,
			batch_size=self.cfg.batch_size, 
			patch_size=self.cfg.patch_size,  
			length=self.cfg.length,  
			num_workers=self.cfg.num_workers,  
			collate_fn=stack_samples,  
			train_split=self.cfg.train_split,
			n_splits=self.cfg.n_splits
			)
                        
                else:
                
                    self.datamodule = KfoldSegmentationDataModule(
			image_dir=self.cfg.image_dir,
			mask_dir=self.cfg.mask_dir,
			batch_size=self.cfg.batch_size, 
			patch_size=self.cfg.patch_size,  
			length=self.cfg.length,  
			num_workers=self.cfg.num_workers,  
			collate_fn=stack_samples,  
			train_split=self.cfg.train_split,
			n_splits=self.cfg.n_splits
			 )
            else:
                if self.cfg.tiles:
                    self.datamodule=TileSegmentationDataModule(
                     image_dir=self.cfg.image_dir,
                     mask_dir=self.cfg.mask_dir,
                     batch_size=self.cfg.batch_size,
                     patch_size=self.cfg.patch_size,
                     num_workers=self.cfg.num_workers,
                     collate_fn=stack_samples,
                     train_split=self.cfg.train_split,
                     )
                else:
                    self.datamodule = SegmentationDataModule(
			    image_dir=self.cfg.image_dir,
			    mask_dir=self.cfg.mask_dir,
			    batch_size=self.cfg.batch_size, 
			    patch_size=self.cfg.patch_size,  
			    length=self.cfg.length,  
			    num_workers=self.cfg.num_workers,  
			    collate_fn=stack_samples,  
			    train_split=self.cfg.train_split,
			    )
            if self.finetune:
                self.task=SegmentationLitModule.load_from_checkpoint(self.checkpoint)
                print(f'weights of {self.checkpoint} loaded successfully')
                
            else:
                self.task = SegmentationLitModule( 
		    model=self.cfg.model,
		    backbone=self.cfg.backbone,
		    weights=self.cfg.weights,
		    in_channels=self.cfg.in_channels,
		    num_classes=self.cfg.num_classes,
		    loss=self.cfg.loss_fn,
		    ignore_index=self.cfg.ignore_index,
		    lr=self.cfg.lr,
		    class_weights=self.cfg.class_weights,
		    patience=self.cfg.patience, 
		    optimizer=self.cfg.optimizer,
		) 
    
        print("Trainer Initialized!") 

    def initialize_detection(self):
        if self.cfg.Kfold:
            self.datamodule=KfoldDetectionDataModule(
                               root_dir=self.cfg.image_dir,
				dataframe=self.cfg.csv_file,
				batch_size=self.cfg.batch_size,
				num_workers=self.cfg.num_workers,  
				train_split=self.cfg.train_split,
				label_map=self.cfg.label_map
				)
				
        else:
        
            self.datamodule=DetectionDataModule(
				root_dir=self.cfg.image_dir,
				dataframe=self.cfg.csv_file,
				batch_size=self.cfg.batch_size,
				num_workers=self.cfg.num_workers,  
				train_split=self.cfg.train_split,
				label_map=self.cfg.label_map
				)
        if self.finetune:
            self.task=DetectionLitModule.load_from_checkpoint(self.checkpoint)
            print(f'weights {self.checkpoint} loaded successfully')
            
        else:
            self.task=DetectionLitModule(
				    model=self.cfg.model,
				    backbone=self.cfg.backbone,
				    weights=self.cfg.weights,
				    in_channels=self.cfg.in_channels,
				    num_classes=self.cfg.num_classes,
				    lr=self.cfg.lr,
				    patience=self.cfg.patience, 
				    optimizer=self.cfg.optimizer,
				    class_weights=self.cfg.class_weights
				    )
        
        
  
    def initialize_classification(self):
        
        if self.cfg.model=='CropDiseaseModel-Resnet34':
        
            self.datamodule = CropDataModule(
                             root_dir=self.cfg.root_dir,
                             batch_size=self.cfg.batch_size,
                             image_size=self.cfg.image_size,
                             num_workers=self.cfg.num_workers,
                             train_split=self.cfg.train_split,
                             )
                             
            if self.finetune:
                self.task=CropDiseaseClassificationLitModule.load_from_checkpoint(self.checkpoint)
            else:
                self.task=CropDiseaseClassificationLitModule( 
                                   model='CropDiseaseModel-Resnet34',
				    weights=self.cfg.weights,
				    in_channels=self.cfg.in_channels,
				    num_crop_classes=self.cfg.num_crop_classes,
				    num_classes=self.cfg.num_disease_classes,
				    lr=self.cfg.lr,
				    optimizer=self.cfg.lr,
				    class_weights=self.cfg.class_weights,
				    freeze_backbone=self.cfg.freeze_backbone,
				    loss=self.cfg.loss ,
				    )
				    
        else:
            if self.cfg.Kfold:
                self.datamodule=KfoldClassificationDataModule(
                     root_dir=self.cfg.root_dir,
                     batch_size=self.cfg.batch_size,
                     image_size=self.cfg.image_size,
                     num_workers=self.cfg.num_workers,
                     train_split=self.cfg.train_split,
                     n_splits=self.n_splits,
                     )
            
            else:
        
                self.datamodule=ClassificationDataModule(
                     root_dir=self.cfg.root_dir,
                     batch_size=self.cfg.batch_size,
                     image_size=self.cfg.image_size,
                     num_workers=self.cfg.num_workers,
                     train_split=self.cfg.train_split,
                     )
				    
            if self.finetune:
                self.task=ClassificationLitModule.load_from_checkpoint(self.checkpoint)
                print(f'weights of {self.checkpoint} loaded successfully')
           
        
            else:
                self.task=ClassificationLitModule(
        	                   model=self.cfg.model,
				    weights=self.cfg.weights,
				    in_channels=self.cfg.in_channels,
				    num_classes=self.cfg.num_classes,
				    lr=self.cfg.lr,
				    patience=self.cfg.patience,
				    loss=self.cfg.loss,
				    optimizer=self.cfg.optimizer,
				    class_weights=self.cfg.class_weights,
				    freeze_backbone=self.cfg.freeze_backbone,
				    )
    
    def initializer(self):
        if self.cfg.task=='segmentation':
            self.initialize_segmentation()
        elif self.cfg.task=='detection':
            self.initialize_detection()
        elif self.cfg.task=='classification':
            self.initialize_classification()
        else:
            raise ValueError(
                f"Task type '{self.cfg.task}' is not valid. "
                "Currently, only supports segmentation, classification and object detection"
            )
            
            
    
        
        
        
        

    def fit(self):
        self.initializer() 
        
        if self.cfg.Kfold:
            for fold in range(self.cfg.n_splits):
                print(f"-------------------Fold-{fold}--------------------")
                if fold!=0:
                    self.datamodule.set_fold(fold)
                
                self.trainer.fit(self.task,self.datamodule)
                if fold ==self.cfg.stop_at_fold:
                    break
        else:
            self.trainer.fit(self.task, self.datamodule) 
            
    
            


            
            

                  


