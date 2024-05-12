
# Autocrop for LexiCam
This repo illustrates how the autocrop of LexiCam's reverse image search works

![Illustration of algorithm](/readme_images/algo_illustration.png "Reverse image search")  

The reverse image search consists of multiple steps involving two round trips.

**1. Autocrop/Bounding box regression**
* Downsampled image is sent to servers
* Autocrop model selects region of interest and returns coordinates

**2. Vectorization of the cropped image (metric learning model)**
* Original image is cropped according to bounding box, downsampled and sent to servers
* Metric learning model calculates embedding from cropped image

**3. Approximate nearest neighbor search in vector database**
* Images which have the most similar vectors are returned as results

If the entire image search takes longer than 1 second, it will be perceived as slow. Since we need some buffer for resizing the images within the mobile app and sending them over the internet (~65KB for 320x480 and ~25KB for 224x224), we should aim for well below 500ms in ideal network conditions. Steps 2 & 3 together take around 200ms. The target latency for the Autocrop microservice should be between 100-200ms.


### Autocrop task
![Crop illustration](/readme_images/crop_illustration_small.png)  
We need to be able to crop two types of objects: paintings and art objects (e.g. vase, jewelry, etc).
The best way to do that is to take a pretrained object detection model and finetune it to the use case. The classification layer will be calculated in vain but should be computationally negligible compared to the feature extraction part. Pytorch has a couple of [pretrained models](https://pytorch.org/vision/stable/models.html#object-detection) available, including information on their performance (mAP) & computational cost (GFLOPS).
FasterRCNN is a state-of-the-art object detection model and allows for different network architectures to be used as feature extractor ("backbone"). Benchmarking different backbones on my Linux laptop (which is not much faster than average server instances) leads to only two contenders: Mobilenet V3 with >500ms and Mobilenet320 V3 with ~120ms. Mobilenet320 works with images where 320 denotes to the shorter side of the image. Using Mobilenet320 has the additional benefit of that the smaller size of the image makes sending it over the internet faster. Mobilenet320 is the best choice atm, more computationally demanding models might make sense in combination with quantization and/or GPU servers.

### Training the model
I use [Optuna](https://optuna.org/) for choosing the appropriate hyperparameters and stratified k-fold Cross validation from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) to split between training and validation dataset.
1. Wide range hyperparameter optimization
2. Narrow range hyperparameter search
3. Training with fixed parameters and many epochs  
At first it was possible to train the model locally, at some point I had to switch to DigitalOcean/Google Colab.

### Check if more data is useful
![graph to test if more data is needed](/readme_images/test_more_data_needed_1050images.png)  
Since obtaining and labeling data is costly in terms of time, we are interested in how much we could improve the model with new data. In general there are diminishing returns to using more training data. With our knowledge of the theoretical curve, we can somewhat extrapolate the benefit of additional data. There is still improvement from using 85% to 100% of the dataset, therefore the model could still benefit significantly from additional data. (Though let's keep in mind that the y-axis showing the loss stops at 0.22.)  
![learning curve](/learning_curves/fastercnnmobile320_1050images_22epochs_3pred.png)  
Plotting the training and validation loss vs epochs we can inspect the learning process. The validation loss seems to oscillate around the training loss, which is an indication that the validation set (which is one third in 3 fold cross validation) fails to be represantative for the entire dataset. Again, additional data might be helpful, so I added another 500 images.  
![new graph to test if more data is needed](/readme_images/test_more_data_needed_1500images.png)  
Running the same experiment with 1500 samples shows diminishing returns, as expected. The decrease in loss between 1050 samples and 1500 samples was about slightly smaller than the decrease from 850 to 1050 samples. Lack of data does not seem to be the biggest problem at this stage.


### Artificial training data (augmented images)
Image augmentation is a way to create new images for training by applying various transformations to original images. [Albumentations](https://albumentations.ai) has a wide range of options. I picked a number of transformations ans settings which I expect to result in realistic images - changing colors, lighting and sharpness substantially and adjusting the crop and perspective slightly.
![Final training curve](/readme_images/augmentation.png)  
This way I increased the number of images for training from about 1500 original images to more than 4700 in total.

### Serving the model in production
[bbRegressionMicroservice](/bbRegressionMicroservice) contains a minimal example of a microservice to serve the model.


#### Model runtime
1. [TorchScript/JIT](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) offers a way to optimize certain models for efficient multithreading during inference with little extra code. Fortunately this works for FasterRCNN and the inference latency decreases by about 40ms.
2. Using [ONNX Runtime](https://onnxruntime.ai) instead of TorchScript we can reduce the average inference latency by another 20-30ms. In order to use ONNX Runtime, we have to convert the model into the .onnx format. [This script](./export_onnx.py) contains the necessary configurations to export the model.

#### Web framework
I chose [FastAPI](https://fastapi.tiangolo.com) to serve the model, because it is easy to use and offers decent resource utilization (async). Compared to web frameworks in system programming languages such as C++, Rust & Go there is performance penalty associated with using a Python web framework. Request will have about 20ms higher latency and most of all a higher variance. That means that in one out of 10 times, a request will take 100ms longer in Python. See [Techempower Benchmark](https://www.techempower.com/) for details. I came to the conclusion that switching to a system programming web framework for serving the model is not worth the effort for the time being.


### Results
![Final training curve](/learning_curves/lr9_30_00004_1.17e-5_0.88.png)  
After trying out many different parameters with Optuna and adjusting the learning rate we get a decent looking learning curve that looks like the model is generalizing well.


### Testing
Testing the model with images that were not being used in the training process to check that it works:
![Test set illustration](/readme_images/testing.png)  
Since I use image augmentation it can happen that two almost identical images are both in the training and test dataset. Therefore I decided to keep a second application test set without augmented images.
As evaluation metric I use intersection-over-union (IOU) and % of misses.  
Test set:               ~75.0% IOU, 3% misses  
Application test set:   ~71.5% IOU, 0% misses

### Possible ways to further improve the model
+ Using a more complex model and keeping the latency low by
  + serving the model on a server with GPUs
  + reducing the computational complexity via [Quantization](https://pytorch.org/docs/stable/quantization.html) (FasterRCNN with Mobilenet was not supported for quantization in PyTorch at the time of writing)
+ Searching in the current dataset for hard cases to guess which additional images would be helpful


