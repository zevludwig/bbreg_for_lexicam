# Autocrop for LexiCam
This repo illustrates how the autocrop of LexiCam's reverse image search works

![Illustration of algorithm](/algo_illustration.png "Reverse image search")

The reverse image search consists of multiple steps involving two round trips.

**1. Autocrop/Bounding box regression**
	+ Downsampled image is sent to servers
	+ Autocrop model selects region of interest and returns coordinates
**2. Vectorization of the cropped image (metric learning model)**
	+ Original image is cropped according to coordinates, downsampled and sent to servers
	+ Metric learning model calculates embedding from cropped image
**3. Approximate kNN search in custom data structure**
	+ Tree of clusters, ordered by similarity (comparable to FAISS)
	+ Built on top of MongoDB, clusters will reside in memory most of the time

If the entire image search takes longer than 1 second, it will be perceived as slow. Since we need some buffer for resizing the images within the mobile app and sending them to the servers (~65KB for 320x480 and ~25KB for 224x224), we should aim for well below 500ms in ideal network conditions. Steps 2 & 3 together take around 200ms. The target latency for the Autocrop microservice should be between 100-200ms.

### Autocrop task 
![Crop illustration](/crop_illustration_small.png)  
We need to be able to crop two types of objects: paintings and art objects (e.g. vase, jewellry, etc)
The best way to do that is to take a pretrained object detection model and finetune it to the use case. The classification layer will be calculated in vain but should be computationally negligible compared to the feature extraction part. Pytorch has a couple of [pretrained models](https://pytorch.org/vision/stable/models.html#object-detection) available, including performance (mAP) & computational cost (GFLOPS).
FasterRCNN is a state-of-the-art object detection model and allows for different network architectures to be used as feature extractor ("backbone"). Benchmarking different backbones on my Linux laptop (which is not much faster than average server instances) leads to only two contenders: Mobilenet V3 with >500ms and Mobilenet320 V3 with ~120ms. Mobilenet320 works with images where 320 denotes to the shorter side of the image. Using Mobilenet320 has the additional benefit of that the smaller size of the image makes sending it over the internet faster. Mobilenet320 is the best choice atm, more computationally demanding models might make sense in combination with quantization or GPU servers.

### Training the model
I use [Optuna](https://optuna.org/) for choosing the appropriate hyperparameters and stratified k-fold Cross validation from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) to split between training and validation dataset.
1. Wide range hyperparameter optimization
2. Narrow range hyperparameter search
3. Training with fixed parameters and many epochs
At first it was possible to train the model locally, later on I had to switch to Google Colab.

### Check if more data is useful
![graph to test if more data is needed](/test_more_data_needed_1050images.png)  
Since obtaining and labeling data is costly in terms of time, I checked how well the model performed using only fractions of the dataset. This way one can better estimate the relationship between model performance and the amount of data. There is still improvement from using 85% to 100% of the dataset, therefore the model would still benefit from additional data. (Though let's keep in mind that the y-axis showing the loss stops at 0.22.)  
![learning curve](/fastercnnmobile320_1050images_22epochs_3pred.png) 
Plotting the training and validation loss vs epochs we can inspect the learning process. The validation loss seems to oscillate around the training loss, which is an indication that the validation set (which is one third in 3 fold cross validation) fails to be represantative for the entire dataset. Again, additional data might be helpful, so I added another 500 images.

### TorchScript/JIT
[PyTorch](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) offers a way to optimize certain models for efficient multithreading during inference. Fortunately this works for FasterRCNN and the inference latency decreases by about 40ms.

### Results
![Final training curve](/last_training.png)
After epoch 12 there is not much visible change, so we can stop here. Comparing the learning curve to the previous learning curve with about 1000 images, it seems a lot more stable.
  
Testing the model with images that were not being used in the training process to check that it works:
![Test set illustration](/test_set.png)

### Possible ways to further improve the model
+ Using a more complex model and keeping the latency low by
  + serving the model on a server with GPUs
  + reducing the computational complexity via [Quantization](https://pytorch.org/docs/stable/quantization.html) 	
+ Expanding the number of parameters available for learning (so far only the head of the net has been retrained, not the feature layers)



