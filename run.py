import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
import os
from skimage import io 
from vismask_resnet import vismask 
import numpy as np

outputimages = "outputimages/"
inputimages = "inputimages/"
imgExt = ".jpg"

imagenames = [fn for fn in os.listdir(inputimages) if fn.endswith(imgExt)]

#Taking batches of 10 images of size 224x224
imgCnt = 125
imgCh = 3
imgH = 224
imgW = 224

#Scaling and normalizing the images to required sizes (mean and std deviation are values required by trained VGG model)
trans = transforms.Compose([transforms.ToPILImage(),
							transforms.Scale(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

imgBatch = torch.Tensor(imgCnt, imgCh, imgH, imgW).cuda()

#------------------------------------------------------------------------------------
def getimages(n, out,fmaps,fmapsmasked):

    h = out.size(2)
    w = out.size(3)

    scalingtr = nn.UpsamplingBilinear2d(size=(h,w))

    #placing all intermediate maps and masks in one big array
    fMapsImg = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w)
    fMapsImgM = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w)

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
        fMapsImg.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmaps[i].float())).data[n]).cuda()
        fMapsImgM.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmapsmasked[i].float())).data[n]).cuda()
    
    return fMapsImg,fMapsImgM

#------------------------------------------------------------------------------------


print (".....Loading model.....")
#model = torch.load('model.pth')
#model.eval()

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,3)
model.cuda()

model.load_state_dict(torch.load('model.pth'))
model.eval()

print ("...Loading Images...")


for i in range (0,imgCnt):
    imgBatch[i,:,:,:] = trans(io.imread(os.path.join(inputimages,imagenames[i])))

imgBatch = Variable(imgBatch.cuda(), volatile = True)

#Obtain visualization mask
vismask, fmaps, fmapsM = vismask(model, imgBatch)

print("....Saving images.....")
to_pil = transforms.ToPILImage()

for i in range(0,imgCnt):
	img1,img2 = getimages(i,vismask,fmaps,fmapsM)
	img1 = to_pil(img1.cpu())
	img2 = to_pil(img2.cpu())

	io.imsave(outputimages + imagenames[i].split(".")[0] + str('maps') + '.png',img1)
	io.imsave(outputimages + imagenames[i].split(".")[0] + str('mask') + '.png',img2)

	io.imsave(outputimages + imagenames[i].split(".")[0] + str('final') + '.png',to_pil(vismask[i].cpu()))

