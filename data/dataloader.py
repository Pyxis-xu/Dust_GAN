import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)
def populate_train_list(orig_images_path, dust_images_path):

	train_list = []
	val_list = []
	
	image_list_dust = glob.glob(dust_images_path + "*.jpg")

	print("Found", len(dust_images_path), "images: ", image_list_dust)

	tmp_dict = {}

	for image in image_list_dust:
		image = image.split("/")[-1]
		key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
		if key in tmp_dict.keys():
			tmp_dict[key].append(image)
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)

	train_keys = []
	val_keys = []

	len_keys = len(tmp_dict.keys())
	for i in range(len_keys):
		if i < len_keys*9/10:
			train_keys.append(list(tmp_dict.keys())[i])
		else:
			val_keys.append(list(tmp_dict.keys())[i])

	for key in list(tmp_dict.keys()):

		if key in train_keys:
			for dust_image in tmp_dict[key]:

				train_list.append([orig_images_path + key, dust_images_path + dust_image])
		else:
			for dust_image in tmp_dict[key]:

				val_list.append([orig_images_path + key, dust_images_path + dust_image])

	random.shuffle(train_list)
	random.shuffle(val_list)

	return train_list, val_list


class dedusting_loader(data.Dataset):

	def __init__(self, orig_images_path, dust_images_path, mode='train'):

		self.train_list, self.val_list = populate_train_list(orig_images_path, dust_images_path)

		if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list),"  {}".format(self.data_list[0]))
		else:
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list),"  {}".format(self.data_list[0]))

		

	def __getitem__(self, index):

		data_orig_path, data_dust_path = self.data_list[index]

		data_orig = Image.open(data_orig_path)
		data_dust = Image.open(data_dust_path)

		data_orig = data_orig.resize((480,640), Image.ANTIALIAS)
		data_dust = data_dust.resize((480,640), Image.ANTIALIAS)

		data_orig = (np.asarray(data_orig)/255.0) 
		data_dust = (np.asarray(data_dust)/255.0)

		data_orig = torch.from_numpy(data_orig).float()
		data_dust = torch.from_numpy(data_dust).float()

		return data_orig.permute(2,0,1).unsqueeze(0), data_dust.permute(2,0,1).unsqueeze(0)

	def __len__(self):
		return len(self.data_list)

