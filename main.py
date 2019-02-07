import torch
import torch.nn as nn
from torchvision import transforms
from data_loader import ImageFolder
import os
import argparse
import mod_vgg
import torchvision.utils as vutils
from tqdm import tqdm
import multiprocessing as ml
import time


def pooling_worker(img, img_mask, p):
	save_path = os.path.join(args.res_dir, p.split(args.dir + '/')[1].split('.')[0])
	create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
	vutils.save_image(img, save_path + '_img.png', normalize=True, padding=0)
	vutils.save_image(img_mask, save_path + '_outputseg.png')
	img[0, :, :] = img[0, :, :] + 10 * img_mask[0, :, :]
	vutils.save_image(img, save_path + '_over.png', normalize=True, padding=0)


def create_path(path):
	if os.path.isdir(path) is False:
		os.makedirs(path)


def main(args):

	print("....Initializing data sampler.....")

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	trans = transforms.Compose([transforms.Resize(224),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								normalize])

	dsets = ImageFolder(os.path.join(args.dir), transform=trans)

	dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.bs,
											   num_workers=10, shuffle=True)

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

	for batch_idx, inp_data in enumerate(tqdm(dset_loaders),1):

		inputs = inp_data[0]
		paths = inp_data[2]
		targets = inp_data[1]

		if args.use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()

		with torch.no_grad():
			outputs, vis_mask = model(inputs)
			output_seg = back_model(vis_mask)

			_, preds = torch.max(outputs, 1)

			# print(preds, targets)
			idx = (preds != targets).nonzero().type(torch.LongTensor)
			idx = idx.view(-1)

			pool = ml.Pool(processes=1)

			inputs, targets, output_seg, idx = inputs.cpu(), targets.cpu(), output_seg.cpu(), idx.cpu()

			for i in idx:
				img, img_mask, p = inputs[i, :, :, :], output_seg[i, :, :, :], paths[i]
				pool.apply_async(pooling_worker, args=(img, img_mask, p))

			pool.close()
			pool.join()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Visual Back Prop')
	parser.add_argument('--cp', '--checkpoint', type=str, default='')
	parser.add_argument('--ms', '--manual_seed', type=int, default=500)
	parser.add_argument('--bs', '--batch_size', type=int, default=10)
	parser.add_argument('--dir', '--directory', default='data')

	args = parser.parse_args()
	torch.manual_seed(args.ms)

	args.use_cuda = torch.cuda.is_available()

	if args.use_cuda:
		torch.cuda.manual_seed_all(args.ms)

	args.res_dir = 'results'

	main(args)
