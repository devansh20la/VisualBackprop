# VisualBackprop

The dependencies required before running the code are as follows:

import torch \
from skimage import io  \
from matplotlib import pyplot as plt 

Steps required before running the code: 

1.  After training a model on the required dataset save it using torch.save as "model.pth" file. (trained model on melanoma images is already present in the folder)
2.  Copy the model in the folder. 
3.  Copy required image in the input folder. (some random images from the dataset for testing are already present in the folder)
4.  Run "run.py" 

The output images will be saved in the output folder. 

The program loads a batch of 10 input images and processes it, if you want to load a batch of more images please edit the code in run.py.

