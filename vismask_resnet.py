import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms,datasets
import numpy as np


#Since pytorch does not save intermediate outputs unlike torch/lua, 
#a new class is created with a newly defined forward function

class myFeatureExtractor(nn.Module):

    def __init__(self, model):
        super(myFeatureExtractor, self).__init__()
        self.submodules = {}

        self.submodules[0] = model.conv1
        self.submodules[1] = model.bn1
        self.submodules[2] = model.relu
        self.submodules[3] = model.maxpool
        self.submodules[4] = model.layer1
        self.submodules[5] = model.layer2
        self.submodules[6] = model.layer3
        self.submodules[7] = model.layer4

    def handlesequential(self,x,module,output):
        i = len(output)

        for name,submodule in module._modules.items():
            x = submodule(x)
            output[i] = x.data.clone()
            i+=1
        
        return x,output 

    def forward(self, x):
        outputs = {}
        i=0
        #iterating over modules in the model
        for name in self.submodules:
            # print type(x)
            module = self.submodules[name]
            print type(module)
            if type(module) == torch.nn.modules.activation.ReLU:
                x = module(x)
                outputs[i] = x.data.clone()
                i+=1

            elif type(module) == torch.nn.modules.container.Sequential:
                x,outputs = self.handlesequential(x,module,outputs)
                i = len(outputs)

            else:
                x = module(x)

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
    print type(imgBatch)
    output = model.forward(imgBatch)

    summation = {}
    fmaps = {}
    fmapsmasked = {}
    sumUp = {}
    to_pil = transforms.ToPILImage()

    for i in range(len(output)-1,-1,-1):

        #sum all feature maps in a lyer
        summation[i] = output[i].sum(1)

        #saving the map (sum of all the feature in a layer)
        fmaps[i]= summation[i].clone()

        #point wise multiplication (multiplying output with the previous layer (backpropagating))
        if i < len(output)-1:
            summation[i] = torch.mul(summation[i],sumUp[i + 1])
            summation[i] = normalization(summation[i])

            # summation[i][summation[i] > 0.25] = 1

        #save the intermediate mask (image obtained by backpropagating at every layer)
        fmapsmasked[i] = summation[i].clone()

        if i > 0:
            if output[i].size() == output[i-1].size():

                #scaling up the feature map using deconvolution operation
                mmUp = nn.Sequential(
                    nn.ConvTranspose2d(1,1,kernel_size=(1,1),stride=(1,1)),
                    nn.ConvTranspose2d(1,1,kernel_size=(1,1),stride=(1,1)),
                    nn.ConvTranspose2d(1,1,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                    nn.ConvTranspose2d(1,1,kernel_size=(1,1),stride=(1,1)),
                    )

                mmUp.cuda()

                for j in range(4):
                    mmUp[j].weight.data.fill_(1)
                    mmUp[j].bias.data.fill_(0)

                sumUp[i] = mmUp.forward(Variable(summation[i], volatile=True)).data.clone()

            else:
                mmUp = nn.Sequential(
                    nn.ConvTranspose2d(1,1, kernel_size=(2,2), stride=(2,2)),
                    nn.ConvTranspose2d(1,1,kernel_size=(1,1),stride=(1,1)),
                    nn.ConvTranspose2d(1,1,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                    nn.ConvTranspose2d(1,1,kernel_size=(1,1),stride=(1,1)),
                    )
                mmUp.cuda()

                for j in range(4):
                    mmUp[j].weight.data.fill_(1)
                    mmUp[j].bias.data.fill_(0)

                sumUp[i] = mmUp.forward(Variable(summation[i],volatile=True)).data.clone()

        else:
            mmUp = nn.ConvTranspose2d(1,1,kernel_size=(7,7),stride=(2,2),padding=(3,3),output_padding=(1,1))
            mmUp.cuda()
            mmUp.weight.data.fill_(1)
            mmUp.bias.data.fill_(0)
            sumUp[i] = (mmUp.forward(Variable(summation[i],volatile=True)).data).clone()

    #normalizing the final mask.
    out = sumUp[i]
    out = normalization(out)

    return out, fmaps, fmapsmasked
