import torch
import rasterio
import os
import geopandas as gpd
from torchgeo.datasets import RasterDataset,BoundingBox,GeoDataset,IntersectionDataset 
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset 
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torchvision
import torchvision.transforms as transforms
import cv2


class MyRasterImage(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    separate_files = False

class MyRasterMask(RasterDataset):
    filename_glob = "*.tif"
    is_image = False
    separate_files = False 
    
    
class ClassificationDataset(Dataset):
    def __init__(self,root_dir=None, transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.classes=os.listdir(self.root_dir)
        self.class_to_idx={class_name: idx for idx, class_name in enumerate(self.classes)}
        self.image_paths=[]
        self.image_labels=[]
        for class_name in self.classes:
            cls_dir=os.path.join(root_dir,class_name)
            if os.path.isdir(cls_dir):
                for image_name in os.listdir(cls_dir):
                    image_path=os.path.join(cls_dir, image_name)
                    self.image_paths.append(image_path)
                    self.image_labels.append(self.class_to_idx[class_name])
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self,idx):
        image_path=self.image_paths[idx]
        image=Image.open(image_path).convert('RGB')
        label=self.image_labels[idx]
        if self.transform:
            image=self.transform(image)
        else:
            image=T.ToTensor()(image)
        return {'image':image, 'label':label} 
        
        
        
       
class DetectionDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None,label_map=None):
        super().__init__()
        self.df = dataframe
        self.image_ids = self.df['image_name'].unique()
        self.image_dir = image_dir
        self.transforms = transform
        self.totensor=A.Compose([ToTensorV2])
        self.label_map=label_map
        self.transform=transforms.Compose([
        			ToTensorV2()])
    def __len__(self):
    
        return len(self.image_ids)

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_name'] == image_id] 
        image_sub=set(records['path'])
        

        image_path = self.image_dir+ image_sub.pop()
        
        image=Image.open(image_path).convert('RGB')
        

        boxes = records[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor([self.label_map[item] for item in records['label_name']], dtype=torch.int64)
    
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        
        
        image=transforms.ToTensor()(image)
        return {'image':image,'target':target}
        
  
        
    
class TileSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

        assert len(self.image_files) == len(self.mask_files) 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)
        mask=mask.squeeze(0).long()

        return {"image": image, "mask": mask}


    
class SegmentationDataset(IntersectionDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        image_dataset = MyRasterImage(paths=image_dir)
        mask_dataset = MyRasterMask(paths=mask_dir)
        super().__init__(image_dataset, mask_dataset)
        self.transform = transform 

    def __getitem__(self, query):
        sample = super().__getitem__(query)
        if self.transform:
            sample = self.transform(sample)
        return sample 
        
        
class SegFormerDataset(IntersectionDataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, transform=None):
        image_dataset = MyRasterImage(paths=image_dir)
        mask_dataset = MyRasterMask(paths=mask_dir)
        super().__init__(image_dataset, mask_dataset)
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __getitem__(self, query):
        sample = super().__getitem__(query)
        image = sample["image"]
        mask = sample["mask"]
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        encoded_inputs = self.feature_extractor(images=image, segmentation_maps=mask, return_tensors="pt")
        image = encoded_inputs["pixel_values"].squeeze(0)
        mask = encoded_inputs["labels"].squeeze(0)
        sample['image']=image
        sample['mask']=mask

        return sample
        
        
        
class CropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.samples=[]
        self.class_to_idx={'crop':{}, 'disease':{}}
        self._prepare_dataset()
    def _prepare_dataset(self):
        disease_index=0
        for crop_idx, crop_name in enumerate(os.listdir(self.root_dir)):
            crop_path=os.path.join(self.root_dir, crop_name)
            if os.path.isdir(crop_path):
                self.class_to_idx['crop'][crop_name]=crop_idx
                for disease_idx, disease_name in enumerate(os.listdir(crop_path)):
                    disease_path=os.path.join(crop_path, disease_name)
                    if os.path.isdir(disease_path):
                        self.class_to_idx['disease'][disease_name]=disease_index
                        for image_name in os.listdir(disease_path):
                             image_path=os.path.join(disease_path, image_name)
                             if image_path.endswith(('jpg', 'jpeg','png')):
                                 self.samples.append((image_path, crop_idx, disease_index))
                    disease_index+=1
        
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
      
        image_path, crop_label, disease_label=self.samples[idx]
        image=Image.open(image_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image, crop_label, disease_label 
