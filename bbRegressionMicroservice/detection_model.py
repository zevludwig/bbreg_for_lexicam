
import glob
from PIL import Image
import pickle
import torch
from torch.jit import trace
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

to_tensor = transforms.ToTensor()
scaler = transforms.Resize((480))

def initialize_detection_model():

    model_dict_path = 'mobilenet_320_full_low_lr9_30_lr0001_m88_04_01_24_new_labels.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, box_detections_per_img=1)
    num_classes = 3

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_dict_path, map_location=device))

    files = ['480_full_40.jpg']#, '20220203_182107.jpg', '20220206_163458.jpg', '20220206_163749.jpg', '20220206_164336.jpg', 'w_small.png']#glob.glob(path)
    test_images = [to_tensor(scaler(Image.open('testfiles/' + file))).to(device) for file in files]

    # jit script
    #model = torch.jit.script(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    out = model([test_images[0]])
    #print(model, type(model))

    print(model(test_images))

    #x = torch.rand((1, 3, 224, 224))
    # traced_m = torch.jit.trace(model, x)
    #f = 'test.pt'
    #torch.jit.save(traced_m, f)
    #loaded_m = torch.jit.load(f)
    #torch.onnx._export(loaded_m, x, 'test.onnx', example_outputs=loaded_m(x))

    #torch.onnx.export(model, [test_images[0]], "bbreg.onnx", verbose=True)

    return model, test_images


#initialize_detection_model()

