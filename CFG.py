
import torch

class Segformer_params:
    task='segmentation'
    def __init__(self):
        self.image_dir=None
        self.mask_dir=None
        self.batch_size=8
        self.patch_size=513
        self.length=1000
        self.lr=1e-3
        self.accelerator='gpu'
        self.loading_device='cuda'
        self.devices=[1]
        self.num_classes=2
        self.num_workers=4
        self.patience=5
        self.train_split=0.8
        self.model='segformer'
        self.ignore_index=None
        self.monitor='val_score'
        self.monitor_mode='min'
        self.optimizer='adam'
        self.finetune=False
        self.checkpoint=None
    def update(self,attr_dict=None):
        if attr_dict:
            for key, value in attr_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(
                	f"Parameter '{key}' is not valid. "
           		 )
            
                
        

class SegParams:
    task='segmentation'
    def __init__(self):
        self.image_dir=None
        self.mask_dir=None
        self.output_dir=None
        self.checkpoint_dir=None
        self.checkpoint_name=None
        self.accelerator='gpu'
        self.loading_device='cuda'
        self.devices=[0]
        self.min_epochs=5
        self.max_epochs=100
        self.tiles=False
        self.n_splits=5 #for kfold #works only for cnn models
        self.Kfold=False
        self.top_k_models=1
        self.earlystopping=True
        self.augmentation=False
        self.lr=1e-3
        self.patience=25
        self.batch_size=8
        self.patch_size=1024
        self.length=500
        self.model='unet' # unet, unet++, deeplabv3, deeplabv3+, linknet, fcn
        self.backbone='resnet34' 
        self.weights=False 
        self.loss_fn='ce' # ce, jaccard, focal, dice, lovasz, tversky,boundary loss
        self.class_weights=torch.tensor([0.2,0.8]) 
        self.optimizer='AdamW' # adam, AdamW, RMSProp, SGD
        self.num_workers=7
        self.in_channels=3
        self.num_classes=2
        self.train_split=0.9
        self.ignore_index=None
        self.monitor='val_loss'
        self.monitor_mode='min'
        self.finetune=False
        self.checkpoint=None #if finetune==True
        self.inputs=None
    def update(self,attr_dict=None):
        if attr_dict:
            for key, value in attr_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(
                	f"Parameter '{key}' is not valid. "
           		 )
        
        
class DetectionParams:
    task='detection'
    def __init__(self):
        self.image_dir=None
        self.csv_file=None
        self.output_dir=None
        self.in_channels=3
        self.checkpoint_dir=None
        self.checkpoint_name=None
        self.accelerator='gpu'
        self.loading_device='cuda'
        self.devices=[1] # [0,1] two devices
        self.min_epochs=5
        self.max_epochs=25
        self.Kfold=False
        self.n_splits=5
        self.stop_at_fold=5
        self.top_k_models=1
        self.earlystopping=True
        self.augmentation=False
        self.lr=5e-5  #5e-5
        self.patience=20
        self.batch_size=4
        self.model='retinanet' # 'faster-rcnn', 'fcos', or 'retinanet
        self.backbone='resnet18' #    'resnet18', resnet34, resnet50, resnet101 resnet152, 'resnext50_32x4d' 'resnext101_32x8d' 'wide_resnet50_2' 'wide_resnet101_2'
        self.weights=True
        self.optimizer='adam' #adam AdamW RMSProp SGD
        self.num_workers=4
        self.num_classes=1
        self.train_split=0.8
        
        self.monitor='val_map'
        self.monitor_mode='max'
        self.label_map=None
        self.class_weights=None #{'classification': 0.4, 'bbox_regression': 0.3, 'objectness': 0.3} 
        self.classification_class_weights= None 
        self.finetune=False
        self.checkpoint=None #if finetune==True
    def update(self,attr_dict=None):
        if attr_dict:
            for key, value in attr_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value) 
                else:
                    raise ValueError(
                	f"Parameter '{key}' is not valid. "
           		 )
        
        
        
class ClassificationParams:
    task='classification'
    def __init__(self):
        self.root_dir=None
        self.image_size=224
        self.output_dir=None
        self.checkpoint_dir=None
        self.checkpoint_name=None
        self.accelerator='gpu'
        self.loading_device='cuda'
        self.devices=[0] 
        self.min_epochs=5
        self.max_epochs=50
        self.top_k_models=1
        self.earlystopping=True
        self.lr=5e-5
        self.patience=5
        self.batch_size=8
        self.model='Resnet34'   #'CropDiseaseModel-Resnet34'
        self.weights=True
        self.optimizer='AdamW'
        self.num_workers=4
        self.train_split=0.8
        self.class_weights=None
        self.in_channels=3
        self.num_classes=132
        self.num_crop_classes=7
        self.num_disease_classes=67
        self.loss='ce'
        self.freeze_backbone=False
        self.Kfold=False
        self.n_splits=5
        self.stop_at_fold=5
        self.monitor='val_AverageAccuracy'
        self.monitor_mode='max'
        self.finetune=False
        self.checkpoint=None #if finetune==True
        
    def update(self,attr_dict=None):
        if attr_dict:
            for key, value in attr_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(
                	f"Parameter '{key}' is not valid. "
           		 )
        


class PredictionParams:
    def __init__(self):
        self.task='classification'
        self.model='CropDiseaseModel-Resnet34'
        self.backbone='resnet34'
        self.in_channels=3
        self.num_classes=131
        self. num_crop_classes=7 # for cropdiseasemodel
        self.num_disease_classes=67 # for cropdiseasemodel
        self.accelerator='cuda'
        self.checkpoint_path=None
        
        self.labels_map=None
        self.image_size=224 # for classification tasks
        self.output_dir=None
        self.save_outputs=True
        self.tile_size=1024
        self.nms_threshold=0.1   #for detection
        self.min_confidence=0.25
        self.inputs=None
        
    def update(self,attr_dict=None):
        if attr_dict:
            for key, value in attr_dict.items():
               if hasattr(self, key):
                   setattr(self, key, value)
               else:
                   raise ValueError(f' Parameter {key} is not valid') 
                   
                    
              
               

 



        
        
        
                
        
