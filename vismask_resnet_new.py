import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms,datasets
import numpy as np
from torchvision.models import resnet


#Since pytorch does not save intermediate outputs unlike torch/lua, 
#a new class is created with a newly defined forward function

class myFeatureExtractor(resnet.ResNet):

    def __init__(self,load=None):
        super(myFeatureExtractor, self).__init__(resnet.Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear(2048,3)
        if load is not None:
            self.load_state_dict(torch.load(load)['model'])

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
        for name,submodule in self._modules.items():
            if name.find('relu') != -1 :
                x = submodule(x)
                outputs[i] = x.data.clone()
                i+=1

            elif name.find('layer') != -1:
                x,outputs = self.handlesequential(x,submodule,outputs)
                i = len(outputs)

            else:
                if name.find('fc') != -1:
                    x = x.view(x.size(0),-1)
                x = submodule(x)

        return outputs

def normalization(tensor):
    omin = tensor.min(2,keepdim=True)[0].min(3,keepdim=True)[0].mul(-1)
    omax = tensor.max(2,keepdim=True)[0].max(3,keepdim=True)[0].add(omin)
    tensor = torch.add(tensor,omin.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
    tensor = torch.div(tensor,omax.expand(tensor.size(0),tensor.size(1), tensor.size(2), tensor.size(3)))
    return tensor

def vismask(imgBatch):

    #running the model on the input batch of images and saving output
    model = myFeatureExtractor('model_best.pth.tar')
    model = model.cuda()
    model.eval()
    model.train(False)

    output = model(imgBatch)
    #output = output.cpu()

    summation = {}
    fmaps = {}
    fmapsmasked = {}
    sumUp = {}
    to_pil = transforms.ToPILImage()

    for i in range(len(output)-1,-1,-1):
        # print(output[i].size(),output[i-1].size())
        #sum all feature maps in a lyer
        summation[i] = output[i].sum(1,keepdim=True)

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
                mmUp = nn.ConvTranspose2d(1,1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
                mmUp.weight.data.fill_(1)
                mmUp.bias.data.fill_(0)

                mmUp.cuda()

                sumUp[i] = mmUp(Variable(summation[i].cuda(), volatile=True)).data.clone()

            else:

                mmUp = nn.ConvTranspose2d(1,1,kernel_size=(6,6),stride=(2,2),padding=(2,2))
                mmUp.cuda()

                mmUp.weight.data.fill_(1)
                mmUp.bias.data.fill_(0)

                sumUp[i] = mmUp(Variable(summation[i].cuda(),volatile=True)).data.clone()

        else:
            mmUp = nn.ConvTranspose2d(1,1,kernel_size=(7,7),stride=(2,2),padding=(3,3),output_padding=(1,1))
            mmUp.cuda()
            mmUp.weight.data.fill_(1)
            mmUp.bias.data.fill_(0)
            sumUp[i] = mmUp(Variable(summation[i].cuda(),volatile=True)).data.clone()
        # print(summation[i].size())

    #normalizing the final mask.
    out = sumUp[i]
    out = normalization(out)

    return out, fmaps, fmapsmasked

