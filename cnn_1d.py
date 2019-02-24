import copy

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from sklearn.model_selection import train_test_split

import pandas as pd

# import numpy as np

class Netraw(nn.Module):
	def __init__(self,input_size = 512, output_size = 10):
		super(Netraw,self).__init__()
		self.input_size = input_size
		self.output_size = output_size
	def conv(self,x,in_size,out_size,stride):
		kernel = 3
		padding = 1
		cv = nn.Conv1d(in_size,out_size,kernel,stride=stride,padding=padding)
		return cv(x)
	def forward(self,x,final_size = 32):
		in_size = 1
		out_size = 8
		size_x = x.size()[2]
		while size_x > final_size:
			x = F.relu(self.conv(in_size,out_size,1))     # conv
			x = F.relu(self.conv(out_size,out_size,2))	 # reduce size
			in_size = out_size
			out_size *= 2
			size_x = x.size()[2]
		return x


class Net(nn.Module):
	def __init__(self,input_size = 28, output_size = 10):
		super(Net,self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.conv1 = nn.Conv1d(1,8,5,stride = 1, padding = 2)
		self.conv2 = nn.Conv1d(8,8,3,stride = 2, padding = 1)
		self.conv3 = nn.Conv1d(8,16,5,stride = 1, padding = 2)
		self.conv4 = nn.Conv1d(16,16,3,stride = 2, padding = 1)
		ls = self.input_size//4*16
		self.fc1 = nn.Linear(ls,50)
		self.fc2 = nn.Linear(50,self.output_size)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		ls = self.input_size//4*16
		x = x.view(-1,ls)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
