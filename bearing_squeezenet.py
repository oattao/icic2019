# bearing
import os
import time
import copy
import pickle

from fcn import aCNN

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

# function
def split_data(dataset,train_portion,val_portion):
	"""Split data by train, validation, test set"""
	num_total = len(dataset)
	num_classes = len(dataset.classes)
	num_samples_class = int(num_total/num_classes)
	train_split = round(num_samples_class*train_portion)
	val_split = round(num_samples_class*val_portion)
	train_idx = list()
	val_idx = list()
	test_idx = list()
	for i in range(num_classes):
		class_indices = np.arange(num_samples_class*i,num_samples_class*(i+1))
		train_choice = np.random.choice(class_indices,size=train_split,replace=False)
		train_idx.extend(train_choice)
		class_indices = list(set(class_indices)-set(train_choice))
		val_choice = np.random.choice(class_indices,size=val_split,replace=False)
		val_idx.extend(val_choice)
		test_idx.extend(list(set(class_indices)-set(val_choice)))
	idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
	return idx


# parameter
idx_save = 'idx.pickle'
datasplit = ['train','val','test']
num_epoch = 200

# prepare data
data_dir = '/media/oattao/Oh/MPaper/conference/icic2019/code/gearbox_img'
data_transform = transforms.Compose([transforms.ToTensor(),
									 transforms.Normalize([0.5,0.5,0.5],
														  [0.5,0.5,0.5])])

image_dataset = datasets.ImageFolder(data_dir,data_transform)
# if os.path.isfile(idx_save):
# 	fp = open(idx_save,'rb')
# 	idx = pickle.load(fp)
# 	fp.close()
# else:
# 	idx = split_data(image_dataset,0.6,0.2)
# 	fp = open(idx_save,'wb')
# 	pickle.dump(idx,fp)
# 	fp.close()

idx = split_data(image_dataset,0.6,0.2)

data_sampler = {x: SubsetRandomSampler(idx[x]) for x in datasplit}
data_loader = {x: torch.utils.data.DataLoader(image_dataset,
											  batch_size = 100,
											  sampler = data_sampler[x])
			   for x in datasplit}
dataset_size = {x: len(idx[x]) for x in datasplit}
print(dataset_size)		   
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
class_name = image_dataset.classes
num_class = len(class_name)

net = models.alexnet(pretrained=True)
for param in net.parameters():
	param.requires_grad = True
n_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(n_features,num_class)

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
exp_lr = lr_scheduler.StepLR(optimizer,step_size = 7, gamma = 0.1)

log_acc = {'train': [], 'val': []}
print('Training...')
best_acc = 0.0
since = time.time()
for epoch in range(num_epoch):
	if (epoch+1)%10 == 0:
		print('Epoch {}/{}'.format(epoch,num_epoch-1))
	for phase in ['train','val']:
		if phase == 'train':
			exp_lr.step()
			net.train()
		else:
			net.eval()
		running_loss = 0.0
		running_correct = 0

		for inputs, labels in data_loader[phase]:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(phase=='train'):
				outputs = net(inputs)
				_, preds = torch.max(outputs,1)
				loss = criterion(outputs,labels)
				if phase == 'train':
					loss.backward()
					optimizer.step()

			running_loss += loss.item()*inputs.size(0)
			running_correct += torch.sum(preds == labels.data)

		epoch_loss = running_loss/dataset_size[phase]
		epoch_acc = running_correct.double()/dataset_size[phase]

		if (epoch+1)%10 == 0:
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
		if phase == 'val' and epoch_acc > best_acc:
			best_acc = epoch_acc
			best_net = copy.deepcopy(net.state_dict())