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
from typing import Any
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from shapely.geometry import box
from rasterio.features import rasterize
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
from torchgeo.trainers import SemanticSegmentationTask,ObjectDetectionTask,ClassificationTask
from torchgeo.datasets.splits import random_bbox_assignment 
from torchgeo.samplers import RandomGeoSampler

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score,MulticlassAUROC,MultilabelFBetaScore,MulticlassFBetaScore
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
from utils import *
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from CustomModels import *



class Loss:
    def __init__(self,class_weights=None,model=None):
        self.loss=0
        self.class_weights=class_weights
        self.model=model
        
        
    def forward(self, loss_dict):
        if self.class_weights:
            if self.model=='fcos':
                loss=self.class_weights['classification']*loss_dict['classification'] + self.class_weights['bbox_regression']* loss_dict['bbox_regression'] + 					              		self.class_weights['objectness']*loss_dict['bbox_ctrness']
            elif self.model=='faster-rcnn':
                loss=self.class_weights['classification']*loss_dict['loss_classifier'] + self.class_weights['bbox_regression']* loss_dict['loss_box_reg'] + 					              		self.class_weights['objectness']*loss_dict['loss_objectness'] 
            elif self.model=='retinanet':
                loss=self.class_weights['classification']*loss_dict['classification'] + self.class_weights['bbox_regression']* loss_dict['bbox_regression'] 
                
                
        else:
            loss=sum(item for item in loss_dict.values())
        return loss
        
        
        
        
class ClassificationLitModule(ClassificationTask):
    def __init__(self,**kwargs):
        if 'optimizer' in kwargs.keys():
            self.optmizer_name=kwargs.pop('optimizer')
        if 'class_weights' in kwargs.keys():
            class_weights=kwargs.pop('class_weights')
        if 'ignore' in kwargs.keys():
            kwargs.pop('ignore')
        super(ClassificationLitModule,self).__init__(**kwargs)
        self.save_hyperparameters()
        self.lr=kwargs['lr']
        self.optimizer_map={
            'SGD': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
            }
        self.schedulers_map=None
        self.losses=None
    def training_step(self, batch: Any, batch_idx:int, dataloader_idx:int=0):
        x=batch['image']
        y=batch['label']
        batch_size=x.shape[0]
        y_hat=self(x)
        loss:Tensor=self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        self.train_metrics(y_hat,y)
        self.log_dict(self.train_metrics, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        return loss
        
    def validation_step(self, batch:Any, batch_idx:int, dataloader_idx:int=0):
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            batch['prediction'] = y_hat.argmax(dim=-1)
            for key in ['image', 'label', 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )
                plt.close()
    def configure_optimizer(self):
        optimizer_class=self.optimizer_map(self.optimizer_name)
        optimzer=optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer



class DetectionLitModule(ObjectDetectionTask):

    def __init__(self, **kwargs):
        class_weights=None
        if 'optimizer' in kwargs.keys():
            self.optimizer_name=kwargs.pop('optimizer')
        if 'class_weights' in kwargs.keys():
            class_weights=kwargs.pop('class_weights')
        
        self.Loss=Loss(class_weights=class_weights,model=kwargs['model'])
        if 'ignore' in kwargs.keys():
            kwargs.pop('ignore')
        super(DetectionLitModule, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = kwargs['lr']
        self.optimizers = {
            'SGD': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }
        self.losses=None
        self.schedulers_map=None
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()
        x=batch['image']
        batch_size = len(x) # we change this line to support variable size inputs
        y = [
            {"boxes": torch.tensor(batch["boxes"][i]), "labels": torch.tensor(batch["labels"][i])}
            for i in range(batch_size)
        ]

        loss_dict = self(x, y)
       
        train_loss: Tensor = self.Loss.forward(loss_dict)
        self.log('train_loss', train_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
        self.log_dict(loss_dict)
        torch.cuda.empty_cache()
        return train_loss
    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
   
        x=batch['image']
        batch_size = len(x)
        
        y = [
            {'boxes': torch.tensor(batch['boxes'][i]),'labels': torch.tensor(batch['labels'][i])}
            for i in range(batch_size)
        ]

        y_hat=self(x)
        
        metrics = self.val_metrics(y_hat, y)
        metrics.pop('val_classes', None)

        self.log_dict(metrics, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_mAP', metrics['val_map'], batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            batch['prediction_boxes'] = [b['boxes'].cpu() for b in y_hat]
            batch['prediction_labels'] = [b['labels'].cpu() for b in y_hat]
            batch['prediction_scores'] = [b['scores'].cpu() for b in y_hat]
            batch['image'] = batch['image'].cpu()
            sample = unbind_samples(batch)[0]
            # Convert image to uint8 for plotting
            if torch.is_floating_point(sample['image']):
                sample['image'] *= 255
                sample['image'] = sample['image'].to(torch.uint8)

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )
                plt.close()
        
    def configure_optimizer(self):
        optimizer_class = self.optimizers.get(self.optimizer_name)
        optimizer = optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer 
        
        


class SegFormerLitModule(SemanticSegmentationTask):
    def __init__(self, **kwargs):
        self.save_hyperparameters()
        if 'optimizer' in kwargs.keys():
            self.optimizer_name=kwargs.pop('optimizer')
        super(SegFormerLitModule,self).__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = kwargs['lr']
      
        
        self.num_classes=self.hparams['num_classes']
        self.ignore_index=self.hparams['ignore_index']
    
        self.optimizers = {
            'SGD': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }
        
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=self.num_classes, ignore_index=self.ignore_index, average='micro'),
            MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index, average='micro')
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self,x,y):
        outputs=self.model(pixel_values=x,labels=y)
        return outputs
        
    def configure_models(self):
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
         num_labels=self.hparams['num_classes'],ignore_mismatched_sizes=True,reshape_last_stage=True)
        

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        batch_size=x.shape[0]
       
        y_hat = self(x,y)
        loss,logits=y_hat[0],y_hat[1]
        #loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.train_metrics(, y)
        #self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        batch_size=x.shape[0]
        y_hat = self(x,y)
        loss, logits=y_hat[0],y_hat[1]
        #loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics,batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        batch_size=x.shape[0]
        y_hat = self(x,y)
        loss,logits = y_hat[0],y_hat[1] 
        self.log('test_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer_class = self.optimizers.get(self.optimizer_name)
        optimizer = optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer







class SegmentationLitModule(SemanticSegmentationTask):
    def __init__(self, **kwargs):
        if 'optimizer' in kwargs.keys():
            #self.optimizer_name=kwargs.pop('optimizer') 
            self.hparams['optimizer']=kwargs.pop('optimizer')
        if 'ignore' in kwargs.keys():
            kwargs.pop('ignore')
        super(SegmentationLitModule, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = kwargs['lr']
        if 'class_weights' in kwargs.keys() and  kwargs['class_weights'] !=None :
            self.hparams['class_weights']=kwargs['class_weights'] 
        self.optimizers = {
            'SGD': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }
           
            

    def forward(self, x):
        return self.model(x) 
        
    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']
        num_filters: int = self.hparams['num_filters']

        model_map = {
            'unet': smp.Unet,
            'unet++': smp.UnetPlusPlus,
            'deeplabv3+': smp.DeepLabV3Plus,
            'deeplabv3': smp.DeepLabV3,
            'linknet': smp.Linknet,
            'fcn': FCN,
        }
        

        if model not in model_map:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'linknet', 'fcn', and other added models."
            )

        if model in ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'linknet']:
            self.model = model_map[model](
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'fcn':
            self.model = model_map[model](
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            self.model = model_map[model](in_channels=in_channels, num_classes=num_classes)

        if model not in ['fcn']:
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams.get('freeze_backbone', False) and model in ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'linknet']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams.get('freeze_decoder', False) and model in ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'linknet']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
                
    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        ignore_index = self.hparams['ignore_index']
        class_weights = self.hparams.get('class_weights') 

        ignore_value = -1000 if ignore_index is None else ignore_index
        classes = [i for i in range(self.hparams['num_classes']) if i != ignore_index]

        loss_map = {
            'ce': nn.CrossEntropyLoss(ignore_index=ignore_value, weight=class_weights),
            'jaccard': smp.losses.JaccardLoss(mode='multiclass', classes=classes),
            'focal': smp.losses.FocalLoss('multiclass', ignore_index=ignore_index, normalized=True),
            'dice': smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index),
            'lovasz': smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index),
            'tversky': smp.losses.TverskyLoss(mode='multiclass', ignore_index=ignore_index),
            'bce': nn.BCEWithLogitsLoss(pos_weight=class_weights),
            'boundary_loss':BoundaryLoss()
        }

        if loss not in loss_map:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently supports 'ce', 'jaccard', 'focal', 'dice', 'lovasz', 'tversky', or 'bce' loss."
            )

        self.criterion = loss_map[loss]
        FCN,

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.classification.MulticlassAccuracy`: Overall accuracy
          (OA) using 'micro' averaging. The number of true positives divided by the
          dataset size. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassJaccardIndex`: Intersection
          over union (IoU). Uses 'micro' averaging. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassPrecision`: Precision using
          'micro' averaging. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassRecall`: Recall using 'micro'
          averaging. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassF1Score`: F1 score using
          'micro' averaging. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassDice`: Dice score using
          'micro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        num_classes: int = self.hparams['num_classes']
        ignore_index: int | None = self.hparams['ignore_index']

        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average='global',
                    average='micro',
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average='micro'
                ),
                MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average='micro'
                ), 
                MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average='micro'
                ),
                MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average='micro'
                ),
            ]
        )

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


    def training_step(self, batch, batch_idx):
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor. 
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        y_hat = torch.argmax(y_hat, dim=1)
        self.train_metrics(y_hat, y) 
        
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x)
       
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        y_hat = torch.argmax(y_hat, dim=1)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            batch['prediction'] = y_hat.argmax(dim=1)
            for key in ['image', 'mask', 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )
                plt.close()

    def configure_optimizers(self):
        optimizer_class = self.optimizers.get(self.hparams['optimizer'])
        optimizer = optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer 
        
        
        
        
class CropDiseaseClassificationLitModule(ClassificationTask):
    def __init__(self,**kwargs):
        if 'optimizer' in kwargs.keys():
            self.optmizer_name=kwargs.pop('optimizer')
        if 'class_weights' in kwargs.keys():
            class_weights=kwargs.pop('class_weights')
        if 'num_crop_classes' in kwargs.keys():
            self.num_crop_classes=kwargs.pop('num_crop_classes')
        if 'num_classes' in kwargs.keys():
         
            self.num_classes=kwargs['num_classes']
        if 'ignore' in kwargs.keys():
            kwargs.pop('ignore')
        self.num_crop_classes=7
        
        
        super(CropDiseaseClassificationLitModule,self).__init__(**kwargs)
        self.save_hyperparameters()
        self.lr=kwargs['lr']
        self.optimizer_map={
            'SGD': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
            }
        self.schedulers_map=None
        self.losses=None
    def configure_models(self): 
        self.model=CropDiseaseModel(num_crop_classes=self.num_crop_classes, num_disease_classes=self.num_classes)
        
    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        if loss == 'ce':
            self.criterion: nn.Module = nn.CrossEntropyLoss(
                weight=self.hparams['class_weights']
            )
        elif loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == 'jaccard':
            self.criterion = JaccardLoss(mode='multiclass')
        elif loss == 'focal':
            self.criterion = FocalLoss(mode='multiclass', normalized=True)
            
        else:
            raise ValueError(f"Loss type '{loss}' is not valid.")

        
    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.classification.MulticlassAccuracy`: The number of
          true positives divided by the dataset size. Both overall accuracy (OA)
          using 'micro' averaging and average accuracy (AA) using 'macro' averaging
          are reported. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassJaccardIndex`: Intersection
          over union (IoU). Uses 'macro' averaging. Higher valuers are better.
        * :class:`~torchmetrics.classification.MulticlassFBetaScore`: F1 score.
          The harmonic mean of precision and recall. Uses 'micro' averaging.
          Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        metrics = MetricCollection(
            {
                'OverallAccuracy': MulticlassAccuracy(
                    num_classes=self.hparams['num_classes'], average='micro'
                ),
                'AverageAccuracy': MulticlassAccuracy(
                    num_classes=self.hparams['num_classes'], average='macro'
                ),
                'JaccardIndex': MulticlassJaccardIndex(
                    num_classes=self.hparams['num_classes']
                ),
                'F1Score': MulticlassFBetaScore(
                    num_classes=self.hparams['num_classes'], beta=1.0, average='micro'
                ),
                'AUC': MulticlassAUROC(
                    num_classes=self.hparams['num_classes'], average='macro'
                ),
                
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def training_step(self, batch: Any, batch_idx:int, dataloader_idx:int=0):
        x, crop_labels, disease_labels=batch[0], batch[1], batch[2]
        batch_size=x.shape[0]
        crop_pred, disease_pred=self(x)
        
        loss1=self.criterion(crop_pred, crop_labels)
        loss2=self.criterion(disease_pred, disease_labels)
        
        loss=loss1+loss2
        self.log('train_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        #self.train_metrics(y_hat,y)
        #self.log_dict(self.train_metrics, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch:Any, batch_idx:int, dataloader_idx:int=0):
        x, crop_labels, disease_labels=batch[0], batch[1], batch[2]
        batch_size=x.shape[0]
        crop_pred, disease_pred=self(x)
        loss1=self.criterion(crop_pred, crop_labels)
        loss2=self.criterion(disease_pred, disease_labels)
        loss:Tensor=loss1+loss2
        self.log('val_loss', loss, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics(disease_pred, disease_labels)
        self.log_dict(self.val_metrics, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizer(self):
        optimizer_class=self.optimizer_map(self.optimizer_name)
        optimzer=optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer    
        
        
 
         
         
         
