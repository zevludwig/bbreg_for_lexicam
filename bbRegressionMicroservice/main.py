#########################################################
## Endpoint fastapi app to select roi bounding box
#########################################################
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, File
import onnxruntime
from starlette.middleware.cors import CORSMiddleware
from torchvision import transforms
#from time import time

session = onnxruntime.InferenceSession('out_dyn.onnx')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://lexicam.app', 'https://www.lexicam.app', 'http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

to_tensor = transforms.ToTensor()


@app.post("/detect")
async def detectImageSurge(image: bytes = File(...)):

    #t0 = time()
    image_PIL = Image.open(BytesIO(image))
    #t1 = time()

    if not image_PIL.mode == 'RGB':
        image_PIL = image_PIL.convert('RGB')

    output = session.run(None, {'input': to_tensor(image_PIL).numpy()})

    #t2 = time()

    bbox = output[0].tolist()

    if bbox:

        bbox[0] = [bbox[0][0]/image_PIL.width, bbox[0][1]/image_PIL.height, bbox[0][2]/image_PIL.width, bbox[0][3]/image_PIL.height ]

        #t3 = time()

        #print(f'pil: ', t1-t0, 'ort trans: ', t2-t1, 'pp: ', t3-t2)
        return {'bbox': bbox[0]}

    else:
        return {'bbox': [0.25, 0.25, 0.75, 0.75]}













