
import glob
from PIL import Image
import pickle
import torch
from torch.jit import trace
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

to_tensor = transforms.ToTensor()
#scaler = transforms.Resize((640, 640))

def initialize_detection_model():

    model_dict_path = 'mobilenet_320_all_15_07_23_gc.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, box_detections_per_img=1)
    num_classes = 3
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    model.load_state_dict(torch.load(model_dict_path, map_location=device))

    # jit tracing not possible
    #path = 'testfiles/*.jpg'
    #files = glob.glob(path)
    #test_images = [to_tensor(scaler(Image.open(file))).to(device) for file in files]
    
    # jit script
    model = torch.jit.script(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    return model
    
    
    

