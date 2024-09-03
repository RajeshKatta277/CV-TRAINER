import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from Inferences import *

from CFG import PredictionParams


class Predictor:
    def __init__(self,task=None,cfg=None):
        self.cfg=cfg
        self.task=self.cfg.task
        self.model=self.cfg.model
        self.backbone=self.cfg.backbone
        self.checkpoint_path=self.cfg.checkpoint_path
        self.accelerator=self.cfg.accelerator
        self.in_channels=self.cfg.in_channels
        self.num_classes=self.cfg.num_classes
        self.image_size=self.cfg.image_size
        self.num_crop_classes=self.cfg.num_crop_classes
        self.num_disease_classes=self.cfg.num_disease_classes
        self.labels_map=self.cfg.labels_map
        self.inference=self.initialize()
        
        
        
        
    def cropdiseasemodel(self):
        return CropDiseaseClassificationInference(model=self.model,checkpoint_path=self.checkpoint_path, in_channels=self.in_channels, num_crop_classes=self.num_crop_classes ,num_disease_classes=self.num_disease_classes,image_size=self.image_size, device=self.accelerator,labels_map=self.labels_map)
    def classification(self):
        if self.model=='CropDiseaseModel-Resnet34':
            return CropDiseaseClassificationInference(model=self.model,checkpoint_path=self.checkpoint_path, in_channels=self.in_channels, num_crop_classes=self.num_crop_classes ,num_disease_classes=self.num_disease_classes,image_size=self.image_size, device=self.accelerator)
            
        return ClassificationInference(model=self.model,checkpoint_path=self.checkpoint_path,in_channels=self.in_channels, num_classes=self.num_classes,image_size=224, accelearator=self.accelerator,labels_map=self.labels_map) 
        
    def segmentation(self):
        return Inference_segmentation(model=self.model,accelerator=self.accelerator,checkpoint=self.checkpoint_path,backbone=self.backbone,in_channels=self.in_channels,num_classes=self.num_classes)
    def detection(self):
        return  InferenceModelLoader(model=self.model,accelerator=self.accelerator,checkpoint_path=self.checkpoint_path,backbone=self.backbone,in_channels=self.in_channels,num_classes=self.num_classes,labels_map=self.labels_map)
    def initialize(self):
        if self.task=='classification':
            return self.classification()
        elif self.task=='detection':
            return self.detection()
        elif self.task=='segmentation':
            return self.segmentation()
        elif self.task=='cropdiseaseclassification':
            return self.cropdiseasemodel()
            
    def predict(self,image=None):
        if self.task=='segmentation':
            return self.inference.predict(image=image, output_dir=self.cfg.output_dir, tile_size=self.cfg.tile_size)
        elif self.task=='detection':
            return self.inference.predict(image=image,output_dir=self.cfg.output_dir,save_result=self.cfg.save_outputs, nms_threshold=self.cfg.nms_threshold,min_confidence=self.cfg.min_confidence)  #returns a dictionary 
        else:
            return self.inference.predict(image=image) 
        
    
        



'''pest_labels_map={0: 'Adristyrannus', 1: 'Aleurocanthus spiniferus', 2: 'alfalfa plant bug', 3: 'alfalfa seed chalcid', 4: 'alfalfa weevil', 5: 'Ampelophaga', 6: 'Aphis citricola Vander Goot', 7: 'Apolygus lucorum', 8: 'army worm', 9: 'asiatic rice borer', 10: 'Bactrocera tsuneonis', 11: 'beet army worm', 12: 'beet fly', 13: 'Beet spot flies', 14: 'beet weevil', 15: 'beetle', 16: 'bird cherry-oataphid', 17: 'black cutworm', 18: 'Black hairy', 19: 'blister beetle', 20: 'bollworm', 21: 'brown plant hopper', 22: 'cabbage army worm', 23: 'cerodonta denticornis', 24: 'Ceroplastes rubens', 25: 'Chlumetia transversa', 26: 'Chrysomphalus aonidum', 27: 'Cicadella viridis', 28: 'Colomerus vitis', 29: 'corn borer', 30: 'corn earworm', 31: 'cutworm', 32: 'Dacus dorsalis(Hendel)', 33: 'Dasineura sp', 34: 'Deporaus marginatus Pascoe', 35: 'english grain aphid', 36: 'Erythroneura apicalis', 37: 'fall armyworm', 38: 'Field Cricket', 39: 'flax budworm', 40: 'flea beetle', 41: 'Cicadellidae', 42: 'Fruit piercing moth', 43: 'Gall fly', 44: 'grain spreader thrips', 45: 'grasshopper', 46: 'green bug', 47: 'grub', 48: 'Icerya purchasi Maskell', 49: 'Indigo caterpillar', 50: 'Jute aphid', 51: 'Jute hairy', 52: 'Jute red mite', 53: 'Jute semilooper', 54: 'Jute stem girdler', 55: 'Jute Stem Weevil', 56: 'Jute stick insect', 57: 'large cutworm', 58: 'Lawana imitata Melichar', 59: 'Leaf beetle', 60: 'legume blister beetle', 61: 'Limacodidae', 62: 'Locust', 63: 'Locustoidea', 64: 'longlegged spider mite', 65: 'Lycorma delicatula', 66: 'lytta polita', 67: 'Mango flat beak leafhopper', 68: 'meadow moth', 69: 'Mealybug', 70: 'Miridae', 71: 'mites', 72: 'mole cricket', 73: 'Nipaecoccus vastalor', 74: 'odontothrips loti', 75: 'oides decempunctata', 76: 'paddy stem maggot', 77: 'Panonchus citri McGregor', 78: 'Papilio xuthus', 79: 'parathrene regalis', 80: 'Parlatoria zizyphus Lucus', 81: 'peach borer', 82: 'penthaleus major', 83: 'Phyllocnistis citrella Stainton', 84: 'Phyllocoptes oleiverus ashmead', 85: 'Pieris canidia', 86: 'Prodenia litura', 87: 'yellow rice borer', 88: 'Yellow Mite', 89: 'yellow cutworm', 90: 'Xylotrechus', 91: 'wireworm', 92: 'whitefly', 93: 'white margined moth', 94: 'white backed plant hopper', 95: 'wheat sawfly', 96: 'wheat phloeothrips', 97: 'wheat blossom midge', 98: 'Viteus vitifoliae', 99: 'Unaspis yanonensis', 100: 'Trialeurodes vaporariorum', 101: 'Toxoptera citricidus', 102: 'Toxoptera aurantii', 103: 'Thrips', 104: 'therioaphis maculata Buckton', 105: 'Tetradacus c Bactrocera minax', 106: 'Termite odontotermes (Rambur)', 107: 'Termite', 108: 'tarnished plant bug', 109: 'Sternochetus frigidus', 110: 'stem borer', 111: 'Spilosoma Obliqua', 112: 'small brown plant hopper', 113: 'sericaorient alismots chulsky', 114: 'Scirtothrips dorsalis Hood', 115: 'sawfly', 116: 'Salurnis marginella Guerr', 117: 'rice water weevil', 118: 'Rice Stemfly', 119: 'rice shell pest', 120: 'rice leafhopper', 121: 'rice leaf roller', 122: 'rice leaf caterpillar', 123: 'rice gall midge', 124: 'Rhytidodera bowrinii white', 125: 'red spider', 126: 'Pseudococcus comstocki Kuwana', 127: 'Potosiabre vitarsis', 128: 'Polyphagotars onemus latus', 129: 'Pod borer', 130: 'aphids'}  ''' 







     




        

              
        
    

 
    
        
        
