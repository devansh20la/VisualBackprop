import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,models,transforms

class myFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(myFeatureExtractor, self).__init__()
        self.features = model.features

    def forward(self, x):
        outputs = []
        for sub in [self.features]:
            for name, module in sub._modules.items():
                x = module(x)
                if type(module) == torch.nn.modules.activation.ReLU:
                    outputs.append(x)
        return outputs

def vismask(model,imgBatch):
    model = myFeatureExtractor(model)
    output = model.forward(imgBatch)
    output.reverse()

    summation = []
    fmaps = []
    fmapsmasked = []
    sumUp = []

    for i in range(0,len(output)):
        #sum all feature maps
        summation.append(output[i].sum(1))
        #calculate dimensions of scaled up map
    #     w = (output[i].size(2) - 1) * 1 + 3
    #     h = (output[i].size(3) - 1) * 1 + 3
        fmaps.append(summation[i].clone())

        #point wise multiplication
        if i > 0:
            summation[i] = summation[i].mul(sumUp[i - 1])
        #save intermediate mask    
        fmapsmasked.append(summation[i].clone())

        if i < len(output) - 2:
            if output[i].size() != output[i+1].size():
    #             adjw = output[i+1].size(2) - w 
    #             adjh = output[i+1].size(3) - h
                mmUp = nn.ConvTranspose2d(1,1,kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1),output_padding=(0,0))
                mmUp.cuda()
                mmUp.weight.data.fill_(1)
                mmUp.bias.data.fill_(0)
                sumUp.append(mmUp.forward(summation[i]))
            else:
                sumUp.append(summation[i])
        else:
            sumUp.append(summation[i])
    out = summation[-1]
    omin = out.min(2)[0].min(3)[0]
    omax = out.max(2)[0].max(3)[0].sub(omin)
    out.sub_(omin.expand(out.size(0), out.size(1), out.size(2), out.size(3)))
    out = torch.div(out,omax.expand(out.size(0),out.size(1), out.size(2), out.size(3)))
    return out, fmaps, fmapsmasked
