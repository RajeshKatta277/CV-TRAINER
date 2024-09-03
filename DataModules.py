
import torch
import torchgeo
from sklearn.model_selection import KFold
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms


from torchgeo.samplers import (
    RandomGeoSampler, 
    RandomBatchGeoSampler, 
    GridGeoSampler
)  
from torchgeo.datamodules import GeoDataModule
from Datasets import *
from torch.utils.data import random_split
from torchgeo.datasets import (
    stack_samples,
    RasterDataset,
    unbind_samples,
)
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader, random_split
import lightning as L
import pandas as pd

def detection_collate_function(batch):
    new_batch={
    		'image':[item['image'] for item in batch],
    		'boxes':[item['target']['boxes'] for item in batch],
    		'labels':[item['target']['labels'] for item in batch],
    		'image_id':[item['target']['image_id'] for item in batch],
    		'area': [item['target']['area'] for item in batch],
    		'iscrowd':[item['target']['iscrowd'] for item in batch]
    		}
    return new_batch 


class DetectionDataModule(L.LightningDataModule):
    def __init__(self,root_dir=None,dataframe=None,batch_size=8,num_workers=4, transform=None,collate_fn=detection_collate_function, train_split=0.8,label_map=None):
        super().__init__()
        self.root_dir=root_dir
        self.dataframe= dataframe
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform
        self.collate_fn=collate_fn
        self.train_split=train_split
        self.label_map=label_map
        
    def setup(self,stage=None):
        self.dataframe=pd.read_csv(self.dataframe)
        self.dataframe.dropna(inplace=True)
        self.dataset=DetectionDataset(dataframe=self.dataframe,image_dir=self.root_dir,transform=self.transform,label_map=self.label_map)
        train_size = int(self.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=self.collate_fn, num_workers=self.num_workers, shuffle=True,drop_last=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,batch_size=self.batch_size,collate_fn=self.collate_fn,num_workers=self.num_workers,shuffle=False,drop_last=False)
        
    


        
       
        
        
class KfoldTileSegmentationDataModule(L.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=8, patch_size=256, num_workers=4, transform=None,collate_fn=stack_samples, train_split=0.8,n_splits=5):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.train_split = train_split
        self.patch_size=patch_size
        self.n_splits=n_splits

    def setup(self, stage=None):
        self.dataset = TileSegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            transform=self.transform
        )
        indices=[x for x in range(len(self.dataset))]
        self.sampler=torch.utils.data.SubsetRandomSampler(indices)
        self.patch_indices=list(self.sampler)
        kf = KFold(n_splits=self.n_splits, shuffle=True,random_state=42)
        self.folds = list(kf.split(self.patch_indices)) 
        self.set_fold(fold=self.current_fold)
    def set_fold(self, fold=None):
        self.current_fold=fold
        train_indices, test_indices=self.folds[fold]
        train_images=[self.patch_indices[i] for i in train_indices]
        val_images=[self.patch_indices[i] for i in test_indices]
        
        self.train_sampler=CustomGeoSampler(self.dataset, train_images)
        self.val_sampler=CustomGeoSampler(self.dataset,val_images)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset,sampler=self.train_sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        
class TileSegmentationDataModule(L.LightningDataModule):
    def __init__(self, image_dir,mask_dir, batch_size=8, patch_size=256, num_workers=4, transform=None, collate_fn=stack_samples, train_split=0.8):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.batch_size=batch_size
        self.patch_size=patch_size
        self.num_workers=num_workers
        self.transform=transform
        self.collate_fn=collate_fn
        self.train_split=train_split
    def setup(self, stage=None):
        self.dataset=TileSegmentationDataset(
             image_dir=self.image_dir,
             mask_dir=self.mask_dir,
             transform=self.transform
         )
         
        train_size=int(self.train_split*len(self.dataset))
        val_size=len(self.dataset)-train_size
        self.train_dataset,self.val_dataset=random_split(self.dataset,[train_size, val_size])
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
         
        
        
        
        
class SegFormerDataModule(GeoDataModule):
    def __init__(self, image_dir, mask_dir, patch_size=512, length=1000, num_workers=4, collate_fn=stack_samples, train_split=0.8, batch_size=8, transform=None):
        super().__init__(self)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.length = length
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.transform = transform
        self.train_split = train_split

    def setup(self, stage=None):
        feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        self.dataset = SegFormerDataset(image_dir=self.image_dir, mask_dir=self.mask_dir, feature_extractor=feature_extractor)
        self.sampler = RandomGeoSampler(self.dataset, size=self.patch_size, length=self.length)
        patch_indices = list(self.sampler)
        train_size = int(self.train_split * len(patch_indices))
        test_size = len(patch_indices) - train_size
        train_patches, test_patches = random_split(patch_indices, [train_size, test_size])
        self.train_sampler = CustomGeoSampler(self.dataset, train_patches)
        self.test_sampler = CustomGeoSampler(self.dataset, test_patches)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.test_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=False, num_workers=self.num_workers)
        
        

         
        
class SegmentationDataModule(GeoDataModule):
    def __init__(self, image_dir, mask_dir, patch_size, length, num_workers, collate_fn, train_split=0.8, batch_size=8, transform=None):
        super().__init__(self)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.length = length
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.transform = transform 
        self.train_split = train_split

    def setup(self, stage=None):
        self.dataset = SegmentationDataset(image_dir=self.image_dir, mask_dir=self.mask_dir, transform=self.transform)
        self.sampler = RandomGeoSampler(self.dataset, size=self.patch_size, length=self.length) 
        patch_indices = list(self.sampler)
        train_size = int(self.train_split * len(patch_indices)) 
        test_size = len(patch_indices) - train_size
        train_patches, test_patches = random_split(patch_indices, [train_size, test_size])
        self.train_sampler = CustomGeoSampler(self.dataset, train_patches)
        self.test_sampler = CustomGeoSampler(self.dataset, test_patches) 
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.test_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=False,num_workers=self.num_workers) 
        
        
        
        

class KfoldSegmentationDataModule(GeoDataModule):
    def __init__(self, image_dir, mask_dir, patch_size, length, num_workers, collate_fn, train_split=0.8, batch_size=8, transform=None, n_splits=5):
        super().__init__(self)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.length = length
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.transform = transform
        self.train_split = train_split
        self.n_splits = n_splits
        self.current_fold = 0
        

    def setup(self, stage=None):
        self.dataset = SegmentationDataset(image_dir=self.image_dir, mask_dir=self.mask_dir, transform=self.transform)
        self.sampler = RandomGeoSampler(self.dataset, size=self.patch_size, length=self.length)  
        self.patch_indices = list(self.sampler) 
        
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        self.folds = list(kf.split(self.patch_indices)) 

        self.set_fold(self.current_fold) 

    def set_fold(self, fold_index):
        self.current_fold = fold_index
        train_indices, test_indices = self.folds[fold_index]

        train_patches = [self.patch_indices[i] for i in train_indices]
        test_patches = [self.patch_indices[i] for i in test_indices]

        self.train_sampler = CustomGeoSampler(self.dataset, train_patches)
        self.test_sampler = CustomGeoSampler(self.dataset, test_patches)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.test_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=False)
        
        
        
        
class KfoldDetectionDataModule(L.LightningDataModule):
    def __init__(self,root_dir=None, dataframe=None, batch_size=8, num_workers=4, transform=None, collate_fn=detection_collate_function, train_split=0.8, label_map=None, n_splits=5):
        super(KfoldDetectionDataModule,self).__init__()
        self.root_dir=root_dir
        self.dataframe=dataframe
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform
        self.collate_fn=collate_fn
        self.train_split=train_split
        self.label_map=label_map
        self.n_splits=n_splits
        self.current_fold=0
        
    def setup(self, stage=None):
        self.dataframe=pd.read_csv(self.dataframe)
        self.dataframe.dropna(inplace=True)
        self.dataset=DetectionDataset(dataframe=self.dataframe, image_dir=self.root_dir, transform=self.transform, label_map=self.label_map)
        indices=[x for x in range(len(self.dataset))]
        self.sampler=torch.utils.data.SubsetRandomSampler(indices)
        self.patch_indices=list(self.sampler)
        kf = KFold(n_splits=self.n_splits, shuffle=True,random_state=42)
        self.folds = list(kf.split(self.patch_indices)) 
        self.set_fold(fold=self.current_fold)
         
        
    def set_fold(self,fold=None):
        self.current_fold=fold
        train_indices, test_indices=self.folds[fold]
        train_images=[self.patch_indices[i] for i in train_indices]
        val_images=[self.patch_indices[i] for i in test_indices]
        
        self.train_sampler=CustomGeoSampler(self.dataset, train_images)
        self.val_sampler=CustomGeoSampler(self.dataset,val_images)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.val_sampler, batch_size=self.batch_size, collate_fn=self.collate_fn)



class ClassificationDataModule(L.LightningDataModule):
    def __init__(self,root_dir, batch_size=8, image_size=224,num_workers=4, train_split=0.8):
        super(ClassificationDataModule, self).__init__()
        self.root_dir=root_dir
        self.batch_size=batch_size
        self.image_size=image_size
        self.num_workers=num_workers
        self.train_split=train_split
        self.transform=transforms.Compose([
                  transforms.Resize((self.image_size,self.image_size)),
                  transforms.ToTensor(),
                  ])
        
    def setup(self, stage=None):
        self.dataset=ClassificationDataset(root_dir=self.root_dir, transform=self.transform)
        train_size = int(self.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)
        



class CustomGeoSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, patches):
        self.data_source = data_source
        self.patches = patches

    def __iter__(self):
        return iter(self.patches)

    def __len__(self):
        return len(self.patches)
                
        
class KfoldClassificationDataModule(L.LightningDataModule):
    def __init__(self,root_dir, batch_size=8, image_size=224,num_workers=4, train_split=0.8,n_splits=5):
        self.root_dir=root_dir
        self.batch_size=batch_size
        self.image_size=image_size
        self.num_workers=num_workers
        self.train_split=train_split
        self.n_splits=n_splits
        self.transform=transforms.Compose([
                  transforms.Resize((self.image_size,self.image_size)),
                  transforms.ToTensor(),
                  ])
        
    def setup(self, stage=None):
        self.dataset=ClassificationDataset(root_dir=self.root_dir, transform=self.transform)
        indices=[x for x in range(len(self.dataset))]
        self.sampler=torch.utils.data.SubsetRandomSampler(indices)
        self.patch_indices=list(self.sampler)
        kf = KFold(n_splits=self.n_splits, shuffle=True,random_state=42)
        self.folds = list(kf.split(self.patch_indices)) 
        self.set_fold(fold=self.current_fold)
         
        
    def set_fold(self,fold=None):
        self.current_fold=fold
        train_indices, test_indices=self.folds[fold]
        train_images=[self.patch_indices[i] for i in train_indices]
        val_images=[self.patch_indices[i] for i in test_indices]
        
        self.train_sampler=CustomGeoSampler(self.dataset, train_images)
        self.val_sampler=CustomGeoSampler(self.dataset,val_images)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        
        
class CropDataModule(L.LightningDataModule):
     def __init__(self, root_dir=None, batch_size=8, image_size=224, num_workers=4, train_split=0.8):
         super(CropDataModule, self).__init__()
         self.root_dir=root_dir
         self.batch_size=batch_size
         self.image_size=image_size
         self.num_workers=num_workers
         self.train_split=train_split
         self.transform=transforms.Compose([
                     transforms.Resize((self.image_size, self.image_size)),
                     transforms.ToTensor(),
                     ])
                     
                     
         
     def setup(self, stage=None):
         self.dataset=CropDataset(root_dir=self.root_dir, transform=self.transform)
         train_size=int(len(self.dataset)*self.train_split)
         val_size=len(self.dataset)-train_size
         self.train_dataset, self.val_dataset=random_split(self.dataset, [train_size, val_size])
         
     def train_dataloader(self):
         return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
     def val_dataloader(self):
         return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
         
         

       
       
    	
