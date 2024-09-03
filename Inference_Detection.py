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

    def __init__(self, model_name, backbone_name, checkpoint_path, device, num_classes,labels_map):
        self.model_name = model_name
        self.backbone_name = backbone_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_classes = num_classes
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
        
        
        
    def predict(self,image=None):
        image=Image.open(image).convert('RGB')
        transform=T.Compose([T.ToTensor()])
        image=transform(image).unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            prediction=self.model(image)
        return image.squeeze(0).permute(1,2,0).cpu(), prediction[0] 
        
        
    def plot(self,image,nms_threshold=0.1):
        img,preds=self.predict(image)
        indices = nms(preds['boxes'], preds['scores'], nms_threshold)
        image_np = img*0.5 + 0.5 
        fig, ax = plt.subplots(1, figsize=(6,4))
        ax.imshow(image_np)
        for idx in indices:
          x_min, y_min, x_max, y_max = preds['boxes'][idx].cpu().numpy()
          labels, scores=preds['labels'][idx].cpu().numpy(), preds['scores'][idx].cpu().numpy()
          print(self.labels_map[int(labels)], scores)
          rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
          ax.add_patch(rect)
          plt.text(x_min,y_min,s=f'{self.labels_map[int(labels)]} confidence:{scores*100:.2f}%',bbox=dict(facecolor='red', alpha=0.5))
        plt.show()
                
        
        
        



img_path='/media/vassarml/HDD/AI_DATABASE/PestDataset/Datasets/pest/Jute semilooper/Aug_Anomis Sabulifera Guenee_8.jpg'


loader = InferenceModelLoader(
model_name='faster-rcnn',
backbone_name='resnet34',
checkpoint_path='/media/vassarml/HDD/Rajesh/test_seg/TRAINER/CheckPoints/pest_detection.ckpt',
device='cuda',
num_classes=131,
labels_map={0: 'Adristyrannus', 1: 'Aleurocanthus spiniferus', 2: 'alfalfa plant bug', 3: 'alfalfa seed chalcid', 4: 'alfalfa weevil', 5: 'Ampelophaga', 6: 'Aphis citricola Vander Goot', 7: 'Apolygus lucorum', 8: 'army worm', 9: 'asiatic rice borer', 10: 'Bactrocera tsuneonis', 11: 'beet army worm', 12: 'beet fly', 13: 'Beet spot flies', 14: 'beet weevil', 15: 'beetle', 16: 'bird cherry-oataphid', 17: 'black cutworm', 18: 'Black hairy', 19: 'blister beetle', 20: 'bollworm', 21: 'brown plant hopper', 22: 'cabbage army worm', 23: 'cerodonta denticornis', 24: 'Ceroplastes rubens', 25: 'Chlumetia transversa', 26: 'Chrysomphalus aonidum', 27: 'Cicadella viridis', 28: 'Colomerus vitis', 29: 'corn borer', 30: 'corn earworm', 31: 'cutworm', 32: 'Dacus dorsalis(Hendel)', 33: 'Dasineura sp', 34: 'Deporaus marginatus Pascoe', 35: 'english grain aphid', 36: 'Erythroneura apicalis', 37: 'fall armyworm', 38: 'Field Cricket', 39: 'flax budworm', 40: 'flea beetle', 41: 'Cicadellidae', 42: 'Fruit piercing moth', 43: 'Gall fly', 44: 'grain spreader thrips', 45: 'grasshopper', 46: 'green bug', 47: 'grub', 48: 'Icerya purchasi Maskell', 49: 'Indigo caterpillar', 50: 'Jute aphid', 51: 'Jute hairy', 52: 'Jute red mite', 53: 'Jute semilooper', 54: 'Jute stem girdler', 55: 'Jute Stem Weevil', 56: 'Jute stick insect', 57: 'large cutworm', 58: 'Lawana imitata Melichar', 59: 'Leaf beetle', 60: 'legume blister beetle', 61: 'Limacodidae', 62: 'Locust', 63: 'Locustoidea', 64: 'longlegged spider mite', 65: 'Lycorma delicatula', 66: 'lytta polita', 67: 'Mango flat beak leafhopper', 68: 'meadow moth', 69: 'Mealybug', 70: 'Miridae', 71: 'mites', 72: 'mole cricket', 73: 'Nipaecoccus vastalor', 74: 'odontothrips loti', 75: 'oides decempunctata', 76: 'paddy stem maggot', 77: 'Panonchus citri McGregor', 78: 'Papilio xuthus', 79: 'parathrene regalis', 80: 'Parlatoria zizyphus Lucus', 81: 'peach borer', 82: 'penthaleus major', 83: 'Phyllocnistis citrella Stainton', 84: 'Phyllocoptes oleiverus ashmead', 85: 'Pieris canidia', 86: 'Prodenia litura', 87: 'yellow rice borer', 88: 'Yellow Mite', 89: 'yellow cutworm', 90: 'Xylotrechus', 91: 'wireworm', 92: 'whitefly', 93: 'white margined moth', 94: 'white backed plant hopper', 95: 'wheat sawfly', 96: 'wheat phloeothrips', 97: 'wheat blossom midge', 98: 'Viteus vitifoliae', 99: 'Unaspis yanonensis', 100: 'Trialeurodes vaporariorum', 101: 'Toxoptera citricidus', 102: 'Toxoptera aurantii', 103: 'Thrips', 104: 'therioaphis maculata Buckton', 105: 'Tetradacus c Bactrocera minax', 106: 'Termite odontotermes (Rambur)', 107: 'Termite', 108: 'tarnished plant bug', 109: 'Sternochetus frigidus', 110: 'stem borer', 111: 'Spilosoma Obliqua', 112: 'small brown plant hopper', 113: 'sericaorient alismots chulsky', 114: 'Scirtothrips dorsalis Hood', 115: 'sawfly', 116: 'Salurnis marginella Guerr', 117: 'rice water weevil', 118: 'Rice Stemfly', 119: 'rice shell pest', 120: 'rice leafhopper', 121: 'rice leaf roller', 122: 'rice leaf caterpillar', 123: 'rice gall midge', 124: 'Rhytidodera bowrinii white', 125: 'red spider', 126: 'Pseudococcus comstocki Kuwana', 127: 'Potosiabre vitarsis', 128: 'Polyphagotars onemus latus', 129: 'Pod borer', 130: 'aphids'}
 )
loader.plot(img_path)
