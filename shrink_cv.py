###################################################
## shrink images with INTER_AREA algo from opencv
###################################################


import cv2 as cv
import glob
from PIL import Image
import os

from pathlib import Path

folders = ['art_amsterdam/E Stedelijk Museum', 'art_amsterdam/H VanGogh Museum', 'art_amsterdam/J Rijksmueum', 'albertina_modern', 'art_images_artworks', 'art_images_manual', 'art_images_manual2', 'art_images_manual3', 'art_images_manual4', 'art_images_new', 'khm_images']

new_folder = 'shrinked320x320'

new_size = 320

all_images = []

for folder in folders:

    extensions = ['.jpg', '.png', '.jpeg']
    images = [x for x in Path(f'{folder}/.').glob('*') if x.suffix.lower() in extensions]
    images = [str(image) for image in images]
    all_images += images

print(f'{len(all_images)} images in folders')

os.mkdir(new_folder)

for img in all_images:
    image = Image.open(img)
    image_name = img.split('.')[0]
    image_name = image_name.replace('.jpg','').replace('.jpeg','').replace('.png','') + '320'
    #print(img)
    
    if image.size[0]<320 or image.size[1]<320:
        new_image = image.resize((320,320))
        new_image.save(f'{new_folder}/{image_name}.jpg', 90)
        
    else:
        new_image = cv.imread(img)
        im_small = cv.resize(new_image,(new_size, new_size), 0, 0, interpolation = cv.INTER_AREA)
        cv.imwrite(f'{new_folder}/{image_name}.jpg', im_small, [cv.IMWRITE_JPEG_QUALITY, 90])
        
