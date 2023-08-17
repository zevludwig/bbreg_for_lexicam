#########################################################
## one endpoint fastapi app to select roi bounding box
#########################################################

from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from fastapi import FastAPI
from fastapi import File, HTTPException

from detection_model import initialize_detection_model


app = FastAPI()

device = torch.device("cpu")
model = initialize_detection_model()

# normalization is done within the model
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


@app.post("/detect")
async def detectImageSurge(image: bytes = File(...)):
       
    image_PIL = Image.open(BytesIO(image))
    
    if not image_PIL.mode == 'RGB':
        image_PIL = image_PIL.convert('RGB')
    
    img_n = to_tensor(image_PIL).to(device)
    
    print('input shape: ', img_n.shape)

    _, output = model([img_n])
    
    bbox = output[0]['boxes'].numpy()
    if bbox.shape[0]>0:
        bbox = (bbox[0]).tolist()
    
    return {'bbox': bbox}

        
    
    





    

