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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


@app.post("/detect")
async def detectImageSurge(image: bytes = File(...)):
       
    image_PIL = Image.open(BytesIO(image))
    

    if not image_PIL.mode == 'RGB':
        image_PIL = image_PIL.convert('RGB')
    
    img_n = to_tensor(image_PIL).unsqueeze(0).to(device)

    #t0=time()
    output = model(img_n)
    
    # select bb
    bbox = output[0]['boxes'].detach().numpy()
    if bbox.shape[0]>0:
        bbox = (bbox[0]).tolist()
    #print(bbox)
    
    return {'bbox': bbox}

        
    
    





    

