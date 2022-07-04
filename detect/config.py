from pydantic import BaseSettings
from typing import List
from os import getenv

class Local(BaseSettings):
    DEBUG: bool = True
    VEC_HOST: str = 'http://127.0.0.1:5001/vec'
    MONGO_HOST: str = 'mongodb://127.0.0.1:27017'
    STATIC_URL: str = 'http://192.168.1.154/static/'
    MONGO_UN: str = 'root'
    MONGO_PW: str = 'ml'
    DB_NAME: str = 'testsurge'
    VEC_COL: str = 'imvec2'
    IMAGE_COL: str = 'image'
    ES_UN: str = 'mvpclient'
    ES_PW: str = 'surge42'
    STATIC_FOLDER: str = '/media/vik/250G flash/imgnet_sample/static_folder/static'  # '/vol/static'
    SIMILARITY_THRESHOLD: int = 10
    USEFUL_THRESHOLD: int = 100000
    NR_ENTITIES: int = 8
    NR_IMAGES_PER_ENTITY: int = 7
    ALLOWED_ORIGINS: List = ['*']#['http://localhost',"http://192.168.1.154","https://192.168.1.154", "https://localhost", "http://\*.lexicam.io", "https://\*.lexicam.io"]


class Production(BaseSettings):
    DEBUG: bool = False
    VEC_HOST: str = getenv('VEC_HOST')
    MONGO_HOST: str = getenv('MONGO_HOST')
    STATIC_URL: str = getenv('STATIC_URL')
    MONGO_UN: str = getenv('MONGO_UN')
    MONGO_PW: str = getenv('MONGO_PW')
    DB_NAME: str = getenv('DB_NAME')
    VEC_COL: str = getenv('VEC_COL')
    IMAGE_COL: str = getenv('IMAGE_COL')
    SIMILARITY_THRESHOLD: int = getenv('SIMILARITY_THRESHOLD')
    USEFUL_THRESHOLD: int = getenv('USEFUL_THRESHOLD')
    NR_ENTITIES: int = getenv('NR_ENTITIES')
    NR_IMAGES_PER_ENTITY: int = getenv('NR_IMAGES_PER_ENTITY')
    ALLOWED_ORIGINS: List = ['*']#['http://localhost',"http://192.168.1.154","https://192.168.1.154", "https://localhost", "http://\*.lexicam.io", "https://\*.lexicam.io"]
    MAX_IMAGE_SIZE: int = getenv('MAX_IMAGE_SIZE')
    MAX_THUMB_SIZE: int = getenv('MAX_THUMB_SIZE')
    ITEMS_PER_PAGE: int = getenv('ITEMS_PER_PAGE')

if getenv('DOCKER_IMAGE')=='True':
    settings = Production()
else:
    settings = Local()



