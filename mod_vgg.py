import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch 
# import data_loader as dl
# from torchvision import transforms
# from torchvision import utils as vutils 
# import random 
# import numpy as np 
# import torch.optim as optim

__all__ = [
	'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
	'vgg19_bn', 'vgg19',
]


model_urls = {
	'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
	'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
	'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
	'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		output = []
		x = self.handlesequential(self.features,x,output)

		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x,output

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def handlesequential(self,layer,x,output):

		# Relu after the last convolution operation
		size = 42

		for i,module in enumerate(layer.children()):
			x = module(x)
			if type(module) == nn.ReLU:
				x2 = x.clone()
				if i != size:
					x2 = x2.detach()
					x2.requires_grad = True
				x2 = x2.sum(1,keepdim=True)
				output.append(x2)
		return x


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn(pretrained=False, **kwargs):
	"""VGG 16-layer model (configuration "D") with batch normalization
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
	return model

class vis_model(nn.Module):

	def __init__(self):
		super(vis_model,self).__init__()

		self.conv1 = nn.ConvTranspose2d(1,1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
		self.conv2 = nn.ConvTranspose2d(1,1,kernel_size=(4,4),stride=(2,2),padding=(1,1))
		self.relu = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m,nn.ConvTranspose2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
				m.weight.requires_grad = False
				m.bias.requires_grad = False

	def normalize(self,img,min,max):
		img.clamp_(min=min, max=max)
		img.add_(-min).div_(max - min + 1e-5)

	def normalization(self,tensor):
		omin = tensor.min(2,keepdim=True)[0].min(3,keepdim=True)[0].mul(-1)
		omax = tensor.max(2,keepdim=True)[0].max(3,keepdim=True)[0].add(omin)
		tensor = torch.add(tensor,omin.expand(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)))
		tensor = torch.div(tensor,omax.expand(tensor.size(0),tensor.size(1), tensor.size(2), tensor.size(3)))
		return tensor

	def forward(self,output):

		summation = output[len(output)-1]
		# summation = summation.sum(1,keepdim=True)

		for i in range(len(output)-1,0,-1):

			if output[i].size() == output[i-1].size():
				step_back = self.conv1(summation)
			else:
				step_back = self.conv2(summation)

			summation = output[i-1]
			# summation = summation.sum(1,keepdim=True)
			summation = torch.mul(summation,step_back)
			summation = self.normalization(summation)

		return summation


if __name__ == '__main__':
	random.seed(123)
	torch.manual_seed(123)
	np.random.seed(123)

	mod = vgg16_bn(pretrained=True)
	mod.eval()
	# quit
	# break
	# mod.classifier[6] = nn.Linear(in_features=4096,out_features=2)
	# print(mod)
	mod_vis = vis_model()
	l1_loss = nn.L1Loss()

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
	dsets = dl.ImageFolder('../data/pascal/val/',transform=trans,training=True)
	dset_loader = torch.utils.data.DataLoader(dsets,batch_size=3,num_workers=2,shuffle=False)
	optimizer = optim.SGD(mod_vis.parameters(),lr=1e-4)

	for i,sample in enumerate(dset_loader):
		inp = sample[0]
		targets = sample[2]
		inputs_seg = sample[1]
		with torch.set_grad_enabled(True):
			result,output = mod(inp)
			out = mod_vis(output)

			temp = out.clone()
			temp = temp.detach()
			temp = torch.mul(temp,inputs_seg)
			loss_2 = l1_loss(out, temp)

			loss = loss_2

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# vutils.save_image(out,'../results/out_{0}_mask.png'.format(i),normalize=True)
		# vutils.save_image(inp,'../results/out_{0}_inp.png'.format(i), normalize=True)
		# vutils.save_image(inputs_seg,'../results/out_{0}_inp_mask.png'.format(i), normalize=True)
		break

	# print(result.topk(5, 1, True, True))
	# print()


