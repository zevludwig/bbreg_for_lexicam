import json
import requests
import glob
import time

files = glob.glob('testfiles/*.jpg')


for image in files:

    results_raw = requests.post('https://detect.lexicam.app/detect', files={'image':('to_detect', open(image,'rb'))})
    results = json.loads(results_raw.content)
    print(results)
    
    time.sleep(1)
