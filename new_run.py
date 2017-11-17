import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
import os
from skimage import io 
from vismask_resnet import vismask as vismask_res
import numpy as np
from data_loader import imageandlabel

outputimages = "outputimages/"

source = "/home/devansh/Melanoma/challenge/challenge_data/test"

#Taking batches of 10 images of size 224x224
imgCnt = 100
imgCh = 3
imgH = 224
imgW = 224

#Scaling and normalizing the images to required sizes (mean and std deviation are values required by trained VGG model)
trans = transforms.Compose([transforms.Scale(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#------------------------------------------------------------------------------------
def getimages(n, out,fmaps,fmapsmasked,imgBatch):

    h = out.size(2)
    w = out.size(3)

    scalingtr = nn.UpsamplingBilinear2d(size=(h,w)).cuda()

    #placing all intermediate maps and masks in one big array
    fMapsImg = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w).cuda()
    fMapsImgM = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w).cuda()

    imgout = torch.Tensor(3,h,w).cuda()

    for i in range(0,len(fmaps)):

        #normalization
        minvalue = fmaps[i][n,0].min()
        maxvalue = fmaps[i][n,0].max()
        fmaps[i][n] = torch.add(fmaps[i][n],-minvalue)
        fmaps[i][n] = torch.div(fmaps[i][n],(maxvalue-minvalue))

        #normalization
        minvalue = fmapsmasked[i][n,0].min()
        maxvalue = fmapsmasked[i][n,0].max()
        fmapsmasked[i][n] = torch.add(fmapsmasked[i][n],-minvalue)
        fmapsmasked[i][n] = torch.div(fmapsmasked[i][n],(maxvalue-minvalue))
        
        #saving the normalized map and mask
        fMapsImg.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmaps[i].float().cuda())).data[n]).cuda()
        fMapsImgM.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmapsmasked[i].float().cuda())).data[n]).cuda()
    
    imgBatch[n,0].data.mul_(0.229).add_(0.485)
    imgBatch[n,1].data.mul_(0.224).add_(0.456)
    imgBatch[n,2].data.mul_(0.225).add_(0.406)

    imgout[0].copy_(imgBatch[n,0].data.add(out[n,0]))
    imgout[1].copy_(imgBatch[n,1].data.add(-out[n,0]))
    imgout[2].copy_(imgBatch[n,2].data.add(-out[n,0]))

    imgout.clamp(0,1)

    return fMapsImg,fMapsImgM,imgout

#------------------------------------------------------------------------------------


print (".....Loading model.....")

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,3)
model.cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

dsets = imageandlabel(source,'image.csv', trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=imgCnt, shuffle=False, num_workers=10)

for data in dset_loaders:

    print ("...Loading Images...")
    imgBatch, label, prediction, path = data['image'], data['label'], data['prediction'], data['path']
    temp = Variable(imgBatch.cuda(), volatile = True)

    #Obtain visualization mask
    vismask, fmaps, fmapsM = vismask_res(model, temp)

    print("....Saving images.....")
    to_pil = transforms.ToPILImage()

    for i in range(0,temp.size()[0]):
    	img1,img2,img3 = getimages(i,vismask,fmaps,fmapsM,temp)
        
    	img1 = to_pil(img1.cpu())
    	img2 = to_pil(img2.cpu())
    	img3 = to_pil(img3.cpu())

        if label[i]==prediction[i]:
            save = 'outputimages/correctlabel/'
        else:
            save = 'outputimages/incorrectlabel/'

        io.imsave(save + path[i].split('.')[0] + str('maps') + '.png',img1)
        io.imsave(save + path[i].split('.')[0] + str('mask') + '.png',img2)
        io.imsave(save + path[i].split('.')[0] + str('overlayedmap') + '.png',img3)
        io.imsave(save + path[i].split('.')[0] + str('final') + '.png',to_pil(vismask[i].cpu()))

