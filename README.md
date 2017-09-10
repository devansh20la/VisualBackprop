# VisualBackprop

The dependencies required before running the code are as follows:

import torch \
from skimage import io  \
from matplotlib import pyplot as plt 

Steps required before running the code: 

1.  After training a model on the required dataset save it using torch.save as "model.pth" file. 
2.  Copy the model in the folder. 
3.  Copy required image in the input folder 
4.  Run "run.py" 

The output images will be saved in the output folder. 

The program loads a batch of 10 input images and processes it, if you want to load a batch of more images please edit the code in run.py.

Some results on skin lesion image classification are shown below:
![isic_0013281](https://user-images.githubusercontent.com/16810812/30251740-eb53144a-9633-11e7-8609-94a4a377c130.png)
![isic_0013281final](https://user-images.githubusercontent.com/16810812/30251741-eb53d3ee-9633-11e7-98ff-d866f449007c.png)
![isic_0013281overlayedmap](https://user-images.githubusercontent.com/16810812/30251739-eb531760-9633-11e7-9002-c6f8726ef5a6.png)
