import torch.nn as nn
import torch.nn.functional as F

class aCNN(nn.Module):
	def __init__(self,input_size=224,output_size=3):
		super(aCNN,self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.conv1 = nn.Conv2d(3,8,3,stride=1,padding=1)
		self.bn1 = nn.BatchNorm2d(8)
		self.bn2 = nn.BatchNorm2d(16)
		self.bn3 = nn.BatchNorm2d(32)
		self.bn4 = nn.BatchNorm2d(64)

		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6 = nn.BatchNorm1d(256)
		self.bn7 = nn.BatchNorm1d(32)

		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(8,16,3,stride=1,padding=1)

		self.conv3 = nn.Conv2d(16,32,3,stride=1,padding=1)
		self.conv4 = nn.Conv2d(32,64,3,stride=1,padding=1)
		self.fc1 = nn.Linear(64*7*7,1024)
		self.fc2 = nn.Linear(1024,256)
		self.fc3 = nn.Linear(256,32)
		self.fc4 = nn.Linear(32,output_size)

	def forward(self,x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.pool(x))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.pool(x))		
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.pool(x))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.pool(x))
		x = F.relu(self.pool(x))
		x = x.view(-1,64*7*7)
		x = F.relu(self.bn5(self.fc1(x)))
		x = F.relu(self.bn6(self.fc2(x)))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

class CNN(nn.Module):
	def __init__(self,input_size=224,output_size=3):
		super(CNN,self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.conv1 = nn.Conv2d(3,8,3,stride=1,padding=1)

		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(8,16,3,stride=1,padding=1)

		self.conv3 = nn.Conv2d(16,32,3,stride=1,padding=1)
		self.conv4 = nn.Conv2d(32,64,3,stride=1,padding=1)
		self.fc1 = nn.Linear(64*7*7,1024)
		self.fc2 = nn.Linear(1024,256)
		self.fc3 = nn.Linear(256,32)
		self.fc4 = nn.Linear(32,output_size)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.pool(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.pool(x))		
		x = F.relu(self.conv3(x))
		x = F.relu(self.pool(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.pool(x))
		x = F.relu(self.pool(x))
		x = x.view(-1,64*7*7)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x