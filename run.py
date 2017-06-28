import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,models,transforms
import os
from skimage import io 
from vismask import vismask 

outputimages = "outputimages/img"
inputimages = "inputimages/"
imgExt = "jpg"

imagenames = [fn for fn in os.listdir(inputimages) if fn.endswith(imgExt)]

#Taking batches of 10 images of size 224x224
imgCnt = 10
imgCh = 3
imgH = 224
imgW = 224

#Scaling and normalizing the images to required sizes (mean and std deviation are values required by trained VGG model)
trans = transforms.Compose([transforms.ToPILImage(),
							transforms.Scale(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# imgBatch = torch.Tensor(imgCnt, imgCh, imgH, imgW)
imgBatch = torch.Tensor(imgCnt, imgCh, imgH, imgW).cuda()

#------------------------------------------------------------------------------------
def getimages(n, out,fmaps,fmapsmasked):

    h = out.size(2)
    w = out.size(3)

    scalingtr = nn.UpsamplingBilinear2d(size=(h,w))

    imgout = torch.Tensor(3, imgH,imgW).cuda()

    #placing all intermediate maps and masks in one big array
    fMapsImg = torch.zeros(1,len(fmaps) * h + (len(fmaps) - 1) * 2, w)
    fMapsImgM = torch.zeros(1,len(fmaps) * h + (len(fmaps) - 1) * 2, w)

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
        fMapsImg.narrow(1,(i)*(h+2),w).copy_(scalingtr(Variable(fmaps[i].float())).data[n]).cuda()
        fMapsImgM.narrow(1,(i)*(h+2),w).copy_(scalingtr(Variable(fmapsmasked[i].float())).data[n]).cuda()

    imgout[0].copy_(imgBatch[n][0].data).add(out[n][0])
    imgout[1].copy_(imgBatch[n][0].data).add(-out[n][0])
    imgout[2].copy_(imgBatch[n][0].data).add(-out[n][0])
    imgout.clamp(0,1)
    
    return imgout,fMapsImg,fMapsImgM

#------------------------------------------------------------------------------------


print (".....Loading model.....")
model = torch.load('model.pth')
# model = models.vgg16(pretrained=True)

print ("...Loading Images...")

for i in range (0,10):
	imgBatch[i,:,:,:] = trans(io.imread(os.path.join(inputimages,imagenames[i])))

imgBatch = Variable(imgBatch, volatile = True)

#Obtain visualization mask
vismask, fmaps, fmapsM = vismask(model, imgBatch)

print("....Saving images.....")
to_pil = transforms.ToPILImage()

for i in range(0,10):
	img1,img2,img3 = getimages(i,vismask,fmaps,fmapsM)
	img1 = to_pil(img1.cpu())
	img2 = to_pil(img2.cpu())
	img3 = to_pil(img3.cpu())

	io.imsave(outputimages + str(3*i+1) + '.png',img1)
	io.imsave(outputimages + str(3*i+2) + '.png',img2)
	io.imsave(outputimages + str(3*i+3) + '.png',img3)

	io.imsave(outputimages + str(100+i) + '.png',to_pil(vismask[i].cpu()))

