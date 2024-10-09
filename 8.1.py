import torch
import numpy as np
# 两个帮助加载数据的工具类
from torch.utils.data import Dataset # 构造数据集，支持索引
from torch.utils.data import DataLoader # 拿出一个mini-batch的一组数据以供训练
import matplotlib.pyplot as plt

# Dataset是一个抽象类，不能被实例化，只能被子类去继承
# 自己定义一个类，继承自Dataset
class DiabetesDataset(Dataset):
	# init()魔法方法：文件小，读取所有的数据，直接加载到内存里
	# 如果文件很大，初始化之后，定义文件列表，再用getitem()读出来
	def __init__(self, filepath):
		# filepath：文件路径
		xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
		self.len = xy.shape[0]  # xy.shape = (n, 9) 0表示知道多少行，即数据集长度
		self.x_data = torch.from_numpy(xy[:, :-1]) # 所有行，前八列
		self.y_data = torch.from_numpy(xy[:, [-1]]) # 所有行，最后一列

	# getitem()魔法方法：实例化类之后，该对象把对应下标的数据拿出来
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index] # 返回的是元组

	# len()魔法方法：使用对象时，可以对数据条数进行返回
	def __len__(self):
		return self.len # 759

dataset = DiabetesDataset('database/diabetes.csv.gz')

# DataLoader是一个加载器，用来帮助我们加载数据的，可以进行对象实例化
# 知道索引，数据长度，就可以自动进行小批量的训练
# dataset：数据集对象 batch_size：小批量的容量 shuffle：数据集是否要打乱
# num_workers：读数据是否用多线程并行读取数据，一般设置4或8，不是越高越好
# 此处因为本人电脑CPU限制，线程设置为0，否则会报错
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True,
						  num_workers=0)

# 定义模型
class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.linear1 = torch.nn.Linear(8, 6)
		self.linear2 = torch.nn.Linear(6, 4)
		self.linear3 = torch.nn.Linear(4, 1)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.sigmoid(self.linear1(x))
		x = self.sigmoid(self.linear2(x))
		x = self.sigmoid(self.linear3(x))
		return x

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 需要放在if/封装一下，否则在windows系统中会报错
if __name__ == '__main__':
	loss_list = []
	epoch_list = []
	# epoch：训练周期，所有样本都参与训练叫做一个epoch
	for epoch in range(1000):
		# lteration：迭代次数 = 样本总数 / mini-batch
		# eg: 10000个样本, batch-size = 1000个, Iteration = 10次
		# 内层每一次跑一个mini-batch
		for i, (inputs, labels) in enumerate(train_loader, 0):
			# enumerate() 用于可迭代/可遍历的数据对象组合为一个索引序列
			# 同时列出数据和数据下标，0表示从索引从0开始
			# inputs，label会自动转换成tensor类型的数据
			y_pred = model(inputs)
			loss = criterion(y_pred, labels)
			# print(epoch, i, loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		loss_list.append(loss.item())
		epoch_list.append(epoch)

	plt.plot(epoch_list, loss_list)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
