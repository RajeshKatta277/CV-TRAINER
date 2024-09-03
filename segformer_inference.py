
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
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation,SegformerConfig
from PIL import Image
import requests
 

 
        
             
class Inference:
    def __init__(self,model=None,accelerator='cpu',checkpoint=None,in_channels=3,num_classes=2):
        self.model_name=model
        self.accelerator=accelerator
        self.checkpoint_path=checkpoint
        self.in_channels=in_channels
        self.classes=num_classes
        
        self.model=self.load_model()
    def load_model(self):
        if self.model_name !='segformer':
            raise ValueError(f"Model {self.model_name} is not supported. Supported models: {list(self.model_map.keys())}")
            
        self.feature_extractor=SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        self.feature_extractor.do_reduce_labels=False
        self.feature_extractor.size=256
        
        
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cuda'))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v
        configuration=SegformerConfig()
        model=SegformerForSemanticSegmentation(configuration)
        model.load_state_dict(new_state_dict) 
        model.eval()
        return model
        

    def create_in_memory_geochip(self, predicted_chip, geotransform, crs):
        photometric = 'MINISBLACK'
        predicted_chip=predicted_chip.unsqueeze(0)
        print(predicted_chip.shape)
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
            inputs = self.feature_extractor(images=image, return_tensors='pt')
            outputs = self.model(**inputs)
            print(image.shape[2:])
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
		logits,
		size=(256,256), # (height, width)
		mode='bilinear',
		align_corners=False
	    )

    
            prediction = upsampled_logits.argmax(dim=1)[0] 
            geotransform = from_origin(bbox.minx, bbox.maxy, pixel_size, pixel_size)
            georref_chips_list.append(self.create_in_memory_geochip(prediction, geotransform, crs)) 
        return georref_chips_list 
      
    def merge_georeferenced_chips(self, chips_list, output_path):
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
          
    def predict(self, file_path, output_dir, tile_size): 
        self.test_dataset = RasterDataset(paths=file_path)
        self.test_sampler = GridGeoSampler(self.test_dataset, size=tile_size, stride=tile_size)
        self.test_loader = DataLoader(self.test_dataset, sampler=self.test_sampler, collate_fn=stack_samples)
        self.pixel_size = self.test_dataset.res
        self.crs = self.test_dataset.crs.to_epsg()
        start = timeit.default_timer() 
        chips_generator = self.georreferenced_chip_generator(self.test_loader, self.crs, self.pixel_size)
        print("The time taken to predict was: ", timeit.default_timer() - start)
        start = timeit.default_timer()  # Measuring the time
        file_name = os.path.join(output_dir, 'merged_prediction.tif')
        self.merge_georeferenced_chips(chips_generator, file_name)
        print("The time taken to generate a georeferenced image and save it was: ", timeit.default_timer() - start) 



inf=Inference(model='segformer',accelerator='cuda',checkpoint='/media/vassarml/HDD/Rajesh/test_seg/TRAINER/CheckPoints/best-checkpoint.ckpt',num_classes=2)
inf.predict(file_path='/media/vassarml/HDD/Rajesh/test_seg/image/a1_updated.tif',output_dir='/media/vassarml/HDD/Rajesh/test_seg/outputs',tile_size=256)
