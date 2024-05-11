
import cv2 as cv
import glob
from math import ceil
from PIL import Image
from PIL.Image import Resampling
import os

from pathlib import Path
from random import choice

from rotation_correction import correct_rotation

folders = ['all_images_full', 'all_images_full/alb']#['roi_test_set_max640']


target = 'shrinked_for_training_bb'#'roi_test_set_max480'

all_images = []

COMPRESSION = 40

SIZE = 480
SIZE_SHORT = 320

def calcNewDimsWH(width, height, maxSize):
    """ calc dims for downscale image according to maximum allowed size """

    if width < height:
        newHeight = maxSize
        ratio = height/maxSize
        newWidth = ceil(width/ratio)
    else:
        newWidth = maxSize
        ratio = width/maxSize
        newHeight = ceil(height/ratio)

    return newWidth, newHeight



for folder in folders:

    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.JPEG', '.PNG']
    images = [x for x in Path(f'{folder}/.').glob('*') if x.suffix in extensions]
    images = [str(image) for image in images]
    all_images += images

print(f'{len(all_images)} images in folders')

for img in all_images:
    image = Image.open(img)
    if not image.mode == 'RGB':
        image = image.convert('RGB')

    image_name = img.split('/').pop()
    image_name = image_name.replace('.jpg','').replace('.jpeg','').replace('.png','').replace('.JPG', '').replace('.JPEG', '').replace('.PNG', '')
    #print(img)
    #new_folder = '_'.join(image_names[:2])
    if not os.path.isdir(target):
        os.mkdir(target)



    if image.size[0]<SIZE_SHORT or image.size[1]<SIZE_SHORT:
        nWl, nHl = calcNewDimsWH(image.size[0], image.size[1], SIZE)
        resample_filter = choice([Resampling.BICUBIC, Resampling.LANCZOS])
        new_image = image.resize((nWl, nHl), resample=resample_filter)
        new_image.save(f'{target}/{image_name}.jpg', format='JPEG', quality=COMPRESSION)

    else:

        #TODO
        # randomly pick among resize methods
        random_number = choice([0,1,2,3,4])

        image, _ = correct_rotation(image) # only manually taken images need correction, those are large

        if random_number==0:
            resample_filter = Resampling.BILINEAR
            image.thumbnail((SIZE, SIZE), resample=resample_filter)
            image.save(f'{target}/{image_name}.jpg', format='JPEG', quality=COMPRESSION)


        elif random_number==1:
            resample_filter = Resampling.BICUBIC
            image.thumbnail((SIZE, SIZE), resample=resample_filter)
            image.save(f'{target}/{image_name}.jpg', format='JPEG', quality=COMPRESSION)


        elif random_number==2:
            resample_filter = Resampling.LANCZOS
            image.thumbnail((SIZE, SIZE), resample=resample_filter)
            image.save(f'{target}/{image_name}.jpg', format='JPEG', quality=COMPRESSION)


        elif random_number==3:
            resample_filter = cv.INTER_AREA
            new_image = cv.imread(img)
            nWl, nHl = calcNewDimsWH(new_image.shape[1], new_image.shape[0], SIZE)
            im_small = cv.resize(new_image,(nWl, nHl), 0, 0, interpolation = resample_filter)
            cv.imwrite(f'{target}/{image_name}.jpg', im_small, [cv.IMWRITE_JPEG_QUALITY, COMPRESSION])


        elif random_number==4:
            resample_filter = cv.INTER_LANCZOS4
            new_image = cv.imread(img)
            nWl, nHl = calcNewDimsWH(new_image.shape[1], new_image.shape[0], SIZE)
            im_small = cv.resize(new_image,(nWl, nHl), 0, 0, interpolation = resample_filter)
            cv.imwrite(f'{target}/{image_name}.jpg', im_small, [cv.IMWRITE_JPEG_QUALITY, COMPRESSION])

