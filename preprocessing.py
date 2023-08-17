######################################################
## shrink image and correct rotation 
## with PIL resize algo
######################################################


import glob
from PIL import Image
import os

from rotation_correction import correct_rotation

from pathlib import Path

extensions = ['.jpg', '.png', '.jpeg']
images = [x for x in Path('.').iterdir() if x.suffix.lower() in extensions]
images = [str(image) for image in images]

for img in images:
    image = Image.open(img)
    image.thumbnail((640,640))
    image_name = img.split('/')[-1]
    image, exif_bytes = correct_rotation(image)
    if exif_bytes:
        image.save('shrinked/'+image_name, exif=exif_bytes)
    else:
        image.save('shrinked/'+image_name, exif=b'')
