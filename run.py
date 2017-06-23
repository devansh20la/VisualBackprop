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
imgFileName = "img"
outFileName = "out"
fmapFileName = "fmap"
maskFileName = "mask"
imgExt = ".jpg"

imagenames = os.listdir(inputimages)

#Taking batches of 10 images of size 224x224
imgCnt = 10
imgCh = 3
imgH = 224
imgW = 224

#Scaling images to required sizes.
trans = transforms.Compose([transforms.ToPILImage(),
							transforms.Scale(256),
							transforms.CenterCrop(224),
							transforms.ToTensor()])

# imgBatch = torch.Tensor(imgCnt, imgCh, imgH, imgW)
imgBatch = torch.Tensor(imgCnt, imgCh, imgH, imgW).cuda()

#------------------------------------------------------------------------------------
def getimages(n, out,fmaps,fmapsmasked):
    h = int(out.size()[2])
    w = int(out.size()[3])

    scalingtr = nn.UpsamplingBilinear2d(size=(h,w))

    imgout = torch.Tensor(3, imgH,imgW).cuda()

    #placing all intermediate maps and masks in one big array
    fMapsImg = torch.ones(1,len(fmaps) * h + (len(fmaps) - 1) * 2, w).cuda()
    fMapsImgM = torch.ones(1,len(fmaps) * h + (len(fmaps) - 1) * 2, w).cuda()

    for i in range(0,len(fmaps)):

        #normalization
        min = fmaps[i][n,0].min()
        max = fmaps[i][n,0].max()
        fmaps[i][n,0] = torch.add(fmaps[i][n,0],-min.expand(fmaps[i][n,0].size()))
        fmaps[i][n,0] = torch.div(fmaps[i][n,0],(max-min).expand(fmaps[i][n,0].size()))

        #normalization
        min = fmapsmasked[i][n,0].min()
        max = fmapsmasked[i][n,0].max()
        fmapsmasked[i][n,0] = torch.add(fmapsmasked[i][n,0],-min.expand(fmapsmasked[i][n,0].size()))
        fmapsmasked[i][n,0] = torch.div(fmapsmasked[i][n,0],(max-min).expand(fmapsmasked[i][n,0].size()))
        
        fMapsImg.narrow(1,(i)*(h+2),w).copy_(scalingtr(fmaps[i].float())[n].data).cuda()
        fMapsImgM.narrow(1,(i)*(h+2),w).copy_(scalingtr(fmapsmasked[i].float())[n].data).cuda()

    imgout[0].copy_(imgBatch[n][0].data).add(out[n][0].data)
    imgout[1].copy_(imgBatch[n][0].data).add(-out[n][0].data)
    imgout[2].copy_(imgBatch[n][0].data).add(-out[n][0].data)
    imgout.clamp(0,1)
    
    return imgout,fMapsImg,fMapsImgM

#------------------------------------------------------------------------------------


print (".....Loading model.....")
model = torch.load('model.pth')
# model = models.vgg16(pretrained=True)

print ("...Loading Images...")

for i in range (0,10):
	imgBatch[i,:,:,:] = trans(io.imread(os.path.join(inputimages,imagenames[i])))

imgBatch = Variable(imgBatch)

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

	io.imsave(outputimages + str(100+i) + '.png',to_pil(vismask[i].data.cpu()))

