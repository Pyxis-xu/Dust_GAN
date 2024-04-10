import torch


from data.dataloader import dedusting_loader
from models.conditional_gan import ConditionalGAN

class opt():
	def __init__(self):
		self.lr=0.0001
		self.beta1=0.5
		self.use_gpu=torch.cuda.is_available()
		self.lambda_A=100
		

def train(model,dataload,start_epoch,end_epoch):

	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	if start_epoch!=0:
		model.load("checkpoints/",start_epoch-1)

	for epoch in range(start_epoch,end_epoch):
		for ite,(ori_img,dust_img_img) in enumerate(dataload):

			ori_img=ori_img.to(device)
			dust_img=dust_img.to(device)
		
			model.train(ori_img,dust_img)

		model.save("checkpoints/",epoch)
		# break12a
	with open('logs.txt', 'a') as f:
		f.write("{}. G_Loss={:.4f}, D_Loss={:.4f}\n".format(epoch, model.loss_G.item(), model.loss_D.item()))


if __name__=="__main__":
	ori_img_path="/root/autodl-tmp/DustGAN/datasets/original/"
	dust_img_path="/root/autodl-tmp/DustGAN/datasets/dust/"

	opt=opt()
	cond_gan=ConditionalGAN(opt)
	dataload=dedusting_loader(ori_img_path,dust_img_path)

	train(cond_gan,dataload,10,100)






