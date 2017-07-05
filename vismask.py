import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,models,transforms
from matplotlib import pyplot as plt

#Since pytorch does not save intermediate outputs unlike torch/lua, 
#a new class is created with a newly defined forward function

class myFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(myFeatureExtractor, self).__init__()
        self.features = model.features

    def forward(self, x):
        outputs = []
        for sub in [self.features]:
            #iteratin gover modules in the model
            for name, module in sub._modules.items():
                x = module(x)
                # Saving results after each ReLU activation layer
                if type(module) == torch.nn.modules.activation.ReLU:
                    outputs.append(x.data)
        return outputs

def normalization(tensor):
    omin = tensor.min(2)[0].min(3)[0].mul(-1)
    omax = tensor.max(2)[0].max(3)[0].add(omin)

    tensor = torch.add(tensor,omin.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
    tensor = torch.div(tensor,omax.expand(tensor.size(0),tensor.size(1), tensor.size(2), tensor.size(3)))
    return tensor

def vismask(model,imgBatch):

    #running the model on the input batch of images and saving output
    model = myFeatureExtractor(model)
    output = model.forward(imgBatch)
    output.reverse()

    summation = []
    fmaps = []
    fmapsmasked = []
    sumUp = []
    to_pil = transforms.ToPILImage()

    for i in range(0,len(output)):

        #sum all feature maps in a lyer
        summation.append(output[i].sum(1))

        #saving the map (sum of all the feature in a layer)
        fmaps.append(summation[i].clone())

        #point wise multiplication (multiplying output with the previous layer (backpropagating))
        if i > 0:
            summation[i] = torch.mul(summation[i],sumUp[i - 1])
            # summation[i] = normalization(summation[i])
            # summation[i][summation[i] > 0.25] = 1

        #save the intermediate mask (image obtained by backpropagating at every layer)
        fmapsmasked.append(summation[i].clone())

        if i < len(output)-1:
            if output[i].size() != output[i+1].size():

                #scaling up the feature map using deconvolution operation
                mmUp = nn.ConvTranspose2d(1,1,kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1),output_padding=(0,0))
                mmUp.cuda()
                mmUp.weight.data.fill_(1)
                mmUp.bias.data.fill_(0)
                sumUp.append(mmUp.forward(Variable(summation[i]),volatile=True).data)

            else:

                #Since output of convolution in VGG16 is of the same size as input no deconvolution is performed if the sizes are the same
                sumUp.append(summation[i].clone())
        else:

            sumUp.append(summation[i].clone())

    #normalizing the final mask.
    out = summation[-1]
    out = normalization(out)

    return out, fmaps, fmapsmasked