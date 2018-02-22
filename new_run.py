import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
import os
from skimage import io 
from vismask_resnet_new import vismask as vismask_res
import numpy as np
from data_loader import imageandlabel
import torchvision.utils as vutils
import pdb

outputimages = "outputimages/"

# source = "/home/devansh/Documents/Melanoma/Classification/data/train/"
source = "inputimages/"
#Taking batches of 10 images of size 224x224
imgCnt = 10
imgCh = 3
imgH = 224
imgW = 224

#Scaling and normalizing the images to required sizes (mean and std deviation are values required by trained VGG model)
trans = transforms.Compose([transforms.Resize(400),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#------------------------------------------------------------------------------------
def getimages(n, out,fmaps,fmapsmasked,imgBatch):
	h = out.size(2)
	w = out.size(3)
	scalingtr = nn.Upsample(size=(h,w)).cuda()
	#placing all intermediate maps and masks in one big array
	fMapsImg = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w).cuda()
	fMapsImgM = torch.ones(1,len(fmaps) * h + (len(fmaps)-1)*20, w).cuda()

	imgout = torch.Tensor(3,h,w).cuda()

	for i in range(0,len(fmaps)):

		# #normalization
		minvalue = fmaps[i][n,0].min()
		maxvalue = fmaps[i][n,0].max()
		fmaps[i][n] = torch.add(fmaps[i][n],-minvalue)
		fmaps[i][n] = torch.div(fmaps[i][n],(maxvalue-minvalue))

		# #normalization
		minvalue = fmapsmasked[i][n,0].min()
		maxvalue = fmapsmasked[i][n,0].max()
		fmapsmasked[i][n] = torch.add(fmapsmasked[i][n],-minvalue)
		fmapsmasked[i][n] = torch.div(fmapsmasked[i][n],(maxvalue-minvalue))

		#saving the normalized map and mask
		fMapsImg.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmaps[i].cuda(),volatile=True)).data[n])
		fMapsImgM.narrow(1,(i)*(h+20),h).copy_(scalingtr(Variable(fmapsmasked[i].cuda(),volatile=True)).data[n])

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


dsets = imageandlabel(source,'visu_train.csv', trans)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=imgCnt, shuffle=False, num_workers=1)

for data in dset_loaders:
	print ("...Loading Images...")
	temp, label, prediction, path = data['image'], data['label'], data['prediction'], data['path']

	if torch.cuda.is_available():
		temp = Variable(temp.cuda(), volatile = True)
	else:
		temp = Variable(temp, volatile = True)

	#Obtain visualization mask
	vismask, fmaps, fmapsM = vismask_res(temp)

	print("....Saving images.....")

	for i in range(0,temp.size()[0]):
		img1,img2,img3 = getimages(i,vismask,fmaps,fmapsM,temp)

		# label[i]==prediction[i]:
		save = 'outputimages/'

		# if label[i]==0 and prediction[i]==0:
		# 	save = save + 'M/TP/'
		# elif label[i]==0 and prediction[i]!=0:
		# 	save = save + 'M/FN/'
		# elif label[i]!=0 and prediction[i]==0:
		# 	save = save + 'M/FP/'
		# elif label[i]!=0 and prediction[i]!=0:
		# 	save = save + 'M/TN/'
		
		# vutils.save_image(img1.cpu(),save + path[i].split('.')[0] + str('maps') + '.eps',normalize=True)
		# vutils.save_image(img2.cpu(),save + path[i].split('.')[0] + str('mask') + '.eps',normalize=True)
		# vutils.save_image(img3.cpu(),save + path[i].split('.')[0] + str('overlayedmap') + '.eps',normalize=True)
		# vutils.save_image(temp[i].data.cpu(),save + path[i].split('.')[0] + '.eps')
		# vutils.save_image(vismask[i].cpu(),save + path[i].split('.')[0] + str('final') + '.eps',normalize=True)
		
		# save = 'outputimages/'
		# if label[i]==1 and prediction[i]==1:
		# 	save = save + 'N/TP/'
		# elif label[i]==1 and prediction[i]!=1:
		# 	save = save + 'N/FN/'
		# elif label[i]!=1 and prediction[i]==1:
		# 	save = save + 'N/FP/'
		# elif label[i]!=1 and prediction[i]!=1:
		# 	save = save + 'N/TN/'
		
		# vutils.save_image(img1.cpu(),save + path[i].split('.')[0] + str('maps') + '.eps',normalize=True)
		# vutils.save_image(img2.cpu(),save + path[i].split('.')[0] + str('mask') + '.eps',normalize=True)
		# vutils.save_image(img3.cpu(),save + path[i].split('.')[0] + str('overlayedmap') + '.eps',normalize=True)
		# vutils.save_image(temp[i].data.cpu(),save + path[i].split('.')[0] + '.eps')
		# vutils.save_image(vismask[i].cpu(),save + path[i].split('.')[0] + str('final') + '.eps',normalize=True)
		
		# save = 'outputimages/'
		# if label[i]==2 and prediction[i]==2:
		# 	save = save + 'SK/TP/'
		# elif label[i]==2 and prediction[i]!=2:
		# 	save = save + 'SK/FN/'
		# elif label[i]!=2 and prediction[i]==2:
		# 	save = save + 'SK/FP/'
		# elif label[i]!=2 and prediction[i]!=2:
		# 	save = save + 'SK/TN/'

#		else:
#			save = 'outputimages/incorrectlabel/'
#			if label[i] == 0:
#				save = save + 'M/'
#			elif label[i] == 1:
#				save = save + 'N/'
#			elif label[i] == 2:
#				save = save + 'SK/'

		vutils.save_image(img1.cpu(),save + path[i].split('.')[0] + str('maps') + '.png',normalize=False)
		vutils.save_image(img2.cpu(),save + path[i].split('.')[0] + str('mask') + '.png',normalize=False)
		vutils.save_image(img3.cpu(),save + path[i].split('.')[0] + str('overlayedmap') + '.png',normalize=False)
		vutils.save_image(temp[i].data.cpu(),save + path[i].split('.')[0] + '.png')
		vutils.save_image(vismask[i].cpu(),save + path[i].split('.')[0] + str('final') + '.png',normalize=False)

