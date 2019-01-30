import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import argparse
import mod_vgg
import torchvision.utils as vutils

def create_path(path):
	if os.path.isdir(path)==False:
		os.makedirs(path)

def main(args):

	print("....Initializing data sampler.....")

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	trans = transforms.Compose([transforms.Resize(224), 
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize])

	dsets = datasets.ImageFolder(os.path.join(args.dir), transform=trans)

	dset_loaders = torch.utils.data.DataLoader(dsets,batch_size= args.bs, num_workers=10, shuffle=True)

	print("....Loading Model.....")
	model = mod_vgg.vgg16_bn(pretrained=True)
	back_model = mod_vgg.vis_model()

	model.eval()
	back_model.eval()

	if args.use_cuda:
		model = model.cuda()
		back_model = back_model.cuda()

		torch.cuda.benchmark = True

		model = nn.DataParallel(model)
		back_model = nn.DataParallel(back_model)

	if args.cp:
		state = torch.load(args.cp)
		model.load_state_dict(state['model'])


	for batch_idx, inp_data in enumerate(dset_loaders,1):

		inputs = inp_data[0]

		if args.use_cuda:
			inputs = inputs.cuda()

		with torch.no_grad():
			outputs, vis_mask = model(inputs)
			output_seg = back_model(vis_mask)

			vutils.save_image(output_seg, os.path.join(args.results_dir,'{0}_outputseg.png'.format(batch_idx)))
			vutils.save_image(inputs, os.path.join(args.results_dir,'{0}_input.png'.format(batch_idx)), normalize=True)

			overlayed = inputs.clone()
			overlayed[:,0,:,:] = overlayed[:,0,:,:] + 10*output_seg[:,0,:,:]
			vutils.save_image(overlayed,os.path.join(args.results_dir,'{0}_overlaid.png'.format(batch_idx)),normalize=True,padding=0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch LUPI Training')
	parser.add_argument('--cp','--checkpoint',type=str,default='')
	parser.add_argument('--ms','--manual_seed',type=int,default=500)
	parser.add_argument('--bs','--batch_size',type=int,default=1)
	parser.add_argument('--dir','--directory',default='data')

	args = parser.parse_args()
	torch.manual_seed(args.ms)

	args.use_cuda = torch.cuda.is_available()

	if args.use_cuda:
		torch.cuda.manual_seed_all(args.ms)

	args.results_dir = os.path.join('results')
	create_path(args.results_dir)

	main(args)
