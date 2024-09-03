import torch
import torch.nn as nn
from CFG import *
from Trainer import *
from Inference import *

class ENGINE:

    def __init__(self, task=None, mode=None, train_param_dict=None, pred_param_dict=None):
        self.task=task
        self.mode=mode
        self.train_param_dict=None
        self.pred_param_dict=None
        self.worker=self.initialize() 
        
    def get_pred_cfg(self):
        param=PredictionParams()
        if self.pred_param_dict:
            param.update(self.pred_param_dict)
        return param
        
    def get_train_cfg(self):
        d={'segmentation':SegParams(), 'detection': DetectionParams(),'segformer':Segformer_params(),'classification': ClassificationParams()}
        if self.task:
            param= d[self.task]
        if self.train_param_dict:
            param.update(attr_dict=self.train_param_dict)
            return param
            
        return param 
        
        
    def initialize_trainer(self):
        
        self.cfg=self.get_train_cfg()
        trainer = Custom_Trainer(task=self.task,cfg=self.cfg)
        return trainer
        
    def initialize_predictor(self):
        self.cfg=self.get_pred_cfg() 
        predictor=Predictor(self.task,cfg=self.cfg)
        return predictor
        
    def initialize(self):
        if self.mode=='training':
            return self.initialize_trainer()
        elif self.mode=='prediction':
            return self.initialize_predictor()
        else:
            raise ValueError (f' mode {self.mode} not defined')
         
        
    def fit(self):
        self.worker.fit()
        
        
    def predict(self,image=None):
        return self.worker.predict(image=image) 
    
        
    def run(self,image=None):
        if self.mode=='training':
            self.fit()
        elif self.mode=='prediction':
            return self.predict(image)  
