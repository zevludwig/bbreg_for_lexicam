
import pickle
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def initialize_detection_model():

    model_dict_path = 'mobilenet_320_all_27_01_22_2.pkl'
    
    model_dict = pickle.load(open(model_dict_path, 'rb'))

    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)

    num_classes = 2
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    model.load_state_dict(model_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)


    return model
