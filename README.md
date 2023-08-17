# Autocrop for LexiCam
As part of LexiCam's AI reverse image search 

![Illustration of algorithm](/algo_illustration.png "Reverse image search")

LexiCam's reverse image search consists of multiple steps involving two round trips.

1. Autocrop/Bounding box regression
	+ Downsampled image is sent to servers
	+ Autocrop model selects region of interest and returns coordinates
2. Vectorization of the cropped image (metric learning model)
	+ Original image is cropped according to coordinates, downsampled and sent to servers
	+ Metric learning model calculates embedding from cropped image
3. Approximate kNN search in custom data structure
	+ Tree of clusters, ordered by similarity (similar to FAISS)
	+ Built on top of MongoDB, clusters will reside in memory most of the time

If the entire image search takes longer than 1 second, it will be perceived as slow. Since we need some buffer for resizing the images within the mobile app and sending them to the servers (~65KB for 320x480 and ~25KB for 224x224), we should aim for well below 500ms in ideal network conditions. Steps 2 & 3 together take around 200ms. The target latency for the Autocrop microservice should be between 100-200ms.

### Autocrop task
![Crop illustration](/crop_illustration_small.svg)
We need to be able to crop two types of objects: paintings and art objects (e.g. vase, jewellry, etc)
The best way to do that is to take a pretrained object detection model and finetune it to the use case. The classification layer will be calculated in vain but should be computationally negligible compared to the feature extraction part. Pytorch has a couple of [pretrained models](https://pytorch.org/vision/stable/models.html#object-detection) available, including performance (mAP) & computational cost (GFLOPS).
FasterRCNN is a state-of-the-art object detection model and allows for different network architectures to be used as feature extractor ("backbone"). 


### Check if more data is useful


### TorchScript/JIT
+ [PyTorch](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) for efficient multithreading during inference 

Testing the model with images that were not being used in the training process to check that it works:
![Test set illustration](/test_set.png)

### Possible ways to further improve the model
+ Using a more complex model and keeping the latency low by
	+ serving the model on a server with GPUs
  + reducing the computational complexity via [Quantization](https://pytorch.org/docs/stable/quantization.html) 	
+ Expanding the number of parameters available for learning (so far only the head of the net has been retrained, not the feature layers)



