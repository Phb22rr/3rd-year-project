import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
from collections import OrderedDict, namedtuple
import torchvision.transforms as transforms
from PIL import Image
import csv
from torch.utils.tensorboard import SummaryWriter

def sanitize_for_windows_path(name):
	invalid_chars = r'<>:"/\\|?*'
	return ''.join('_' if c in invalid_chars else c for c in name)

img1_directory = "data/img1.npz"
training_directory = "data/15VariableLabel.npz"
testing_directory = "data/2nd.npz"

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#class SiameseNetworkDataset():
	#def __init__(self, npz_file = None, transform = None):
		#self.data = np.load(npz_file)
		#self.images = self.data['images']
		#self.transform = self.data['labels']
		#self.transform = transform

	#def get__item__(self, index):
		#img = self.images[index]
		#label = self.labels[index]
		#img = Image.fromarray(img)
		#return img, torch.tensor(label, dtype= torch.float32)

	#def __len__(self):
		#return len(self.images)

class SiameseNetworkDataset(torch.utils.data.Dataset):
	def __init__(self, img1_dir, img2_dir, transform = None):
		self.img1_data = np.load(img1_dir)
		self.img1_images = self.img1_data['images']
		self.img1_labels = self.img1_data['labels']

		self.img2_data = np.load(img2_dir)
		self.img2_images = self.img2_data['images']
		self.img2_labels = self.img2_data['labels']
		self.transform = transform

	def __getitem__(self, index):
		random_index = np.random.choice(len(self.img1_images))
		random_num = random_index
		img1 = Image.fromarray(self.img1_images[random_num]).convert("L")
		img2 = self.img2_images[index]
		img2 = Image.fromarray(img2).convert("L")

		#if self.img1_labels[random_num] == self.img2_labels[index]:
		#	label = 0
		#else:
		#	label = 1
		label = abs(self.img1_labels[random_num] - self.img2_labels[index])

		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
		label = torch.tensor(float(label), dtype=torch.float32).squeeze(0)
		return img1, img2, label

	def __len__(self):
		return len(self.img2_images)

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork,self).__init__()
		self.cnn1 = nn.Sequential(nn.Conv2d(1, 96,
											kernel_size=11, stride = 1),
								  nn.ReLU(inplace = True),
								  nn.LocalResponseNorm(5, alpha = 0.0001, beta = 0.75, k=2),
								  nn.MaxPool2d(3, stride=2),
								  nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
								  nn.ReLU(inplace=True),nn.LocalResponseNorm(5, alpha=0.001, beta=0.75, k=2),
								  nn.MaxPool2d(3, stride=2), nn.Dropout2d(p=0.3),
								  nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
								  nn.ReLU(inplace=True),
								  nn.MaxPool2d(3, stride = 2), nn.Dropout2d(p=0.3), )

		self.fc1 = nn.Sequential(nn.Linear(43264,1024), nn.ReLU(inplace=True),
								 nn.Dropout(p=0.5),
								 nn.Linear(1024, 128),
								 nn.ReLU(inplace=True),
								 nn.Linear(128, 1))
	def forward(self, img1, img2):
		x1 = self.cnn1(img1)
		x2 = self.cnn1(img2)
		x1 = x1.view(x1.size(0), -1)
		x2 = x2.view(x2.size(0), -1)
		distance = torch.abs(x1 - x2)
		score = self.fc1(distance)
		output = 1-torch.sigmoid(score)
		return output

class ContrastiveLoss(torch.nn.Module):
	def __init__(self):
		super(ContrastiveLoss, self).__init__()
		self.bce = nn.BCELoss()
	def forward(self, output, label):
		loss_contrastive= self.bce(output, label)
		return loss_contrastive

class RunBuilder():
	@staticmethod
	def get_runs(params):
		Run = namedtuple('Run', params.keys())
		runs = []
		for v in product(*params.values()):
			runs.append(Run(*v))
		return runs

class RunManager():
	def __init__(self):
		self.epoch_count = 0
		self.epoch_loss = 0
		self.epoch_num_correct = 0
		self.epoch_start_time = None

		self.run_params = None
		self.run_count = 0
		self.run_data = []
		self.run_start_time = None

		self.network = None
		self.loader = None
		self.tb = None

	def begin_run(self, run, network, loader):
		self.run_start_time = time.time()
		self.run_params = run
		self.run_count += 1
		self.network = network
		self.loader = loader
		safe_run_name = sanitize_for_windows_path(str(run))
		self.tb = SummaryWriter(comment=f'-{safe_run_name}')

		# Ensure proper indentation here
		data_sample = next(iter(self.loader))  # Adjust based on actual output

		# Unpack properly
		#characteristics, labels = data_sample[:2]

	def end_run(self):
		self.tb.close()
		self.epoch_count = 0

	def begin_epoch(self):
		self.epoch_start_time = time.time()
		self.epoch_count += 1
		self.epoch_loss = 0
		self.epoch_num_correct = 0

	def end_epoch(self):
		epoch_duration = time.time() - self.epoch_start_time
		run_duration = time.time() - self.run_start_time

		#loss = self.epoch_loss / len(self.loader.dataset)
		#accuracy = self.epoch_num_correct / len(self.loader.dataset)

		#self.tb.add_scalar('Loss', loss, self.epoch_count)
		#self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

	def _get_num_correct(self, preds, labels):
		pred_int = preds >= 0.5
		return torch.sum(torch.abs(pred_int.float() - labels)).item() / len(labels)

	def track_loss(self, loss, batch):
		self.epoch_loss += loss.item() * batch[0].shape[0]

	def track_num_correct(self, preds, labels):
		self.epoch_num_correct += self._get_num_correct(preds, labels)

	def inform(self, discrete_n):
		if self.epoch_count % discrete_n == 0:
			print(self.epoch_count, ' ', self.run_count)

	def save_checkpoint(self, model, optimizer, epoch, filename="checkpoint.pth"):
		print(f"Saving checkpoint at epoch {epoch}")
		torch.save(
			{'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), },
			filename)

	#def validate(self, model, validation_dataloader):
		#model.eval()
		#correct = 0
		#total = 0
		#loss_history = []
		#with torch.no_grad():
			#for data in validation_dataloader:
				#img1, img2, label = data
				#img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
				#output = model(img1,img2).squeeze(1)
				#loss_contrastive = criterion(output,label)
				#loss_history.append(loss_contrastive.item())
				#predictions = (output >0.5).float()
				#correct += (predictions == label).sum().item()
				#total += label.size(0)
		#accuracy = correct / total
		#avg_loss = sum(loss_history)/len(loss_history)
		#return accuracy, avg_loss

params = OrderedDict(lr=[0.05, 0.1],
					 batch_size = [4, 16, 32, 64],
					 number_epochs=[100],
					 op=[torch.optim.SGD])
model = SiameseNetwork()
m=RunManager()
b=RunBuilder.get_runs(params)
for run in b:
	if run.op == torch.optim.SGD:
		optimizer = run.op(model.parameters(), lr=run.lr, weight_decay=0.0005)
	else:
		optimizer = run.op(model.parameters(), lr=run.lr, eps=1e-8, weight_decay=0.0005)

siamese_dataset = SiameseNetworkDataset(img1_dir=img1_directory,
										img2_dir = training_directory,
										transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]))
train_dataloader = DataLoader(siamese_dataset,
							  shuffle=True,
							  num_workers=0,
							  batch_size=run.batch_size,
							  pin_memory = True)
#val_dataset = SiameseNetworkDataset(img1_directory,testing_directory,
                                    #transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()]))
#validation_dataloader = DataLoader(val_dataset,shuffle = False,num_workers = 0,batch_size = run.batch_size,pin_memory = True)

def train(net, train_loader, criterion, optimizer, device, number_epochs = run.number_epochs):
	total_batches_seen = 1
	total_batches = len(train_dataloader)*number_epochs
	net.to(device)
	epoch_number = 1
	counter = []
	loss_history = []
	epoch_acc = []
	optimizer = run.op(net.parameters(),lr=run.lr)
	m.begin_run(run, SiameseNetwork(), train_loader)
	for epoch in range(number_epochs):
		total_samples = 0
		net.train()
		running_loss = 0.0
		correct = 0
		correct01 = 0
		m.begin_epoch()
		for batch_idx, (img1, img2, label) in enumerate(train_dataloader):
			current_iter = epoch*len(train_dataloader) + batch_idx + 1
			img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
			optimizer.zero_grad()
			output = net(img1, img2).squeeze(1)
			loss_contrastive = criterion(output, label)
			loss_contrastive.backward()
			optimizer.step()
			running_loss += loss_contrastive.item()
			predicted = torch.round(output*10)/10
			label_rounded = torch.round(label*10)/10
			correct += (predicted == label_rounded).sum().item()
			predicted01 = torch.round(output)
			label01 = torch.round(label)
			correct01 += (predicted01 == label01).sum().item()
			total_samples +=label.size(0)

			m.track_loss(loss_contrastive, (img1, img2, label))
			m.track_num_correct(predicted, label)

			if total_batches_seen % 100 ==0:
				end_batch = time.time()
				time_training = end_batch - start_whole_run

				progress_fraction = current_iter / total_batches
				percent_complete = ((total_parameters_trained+progress_fraction-1)/total_parameters) * 100

				percentage_left = 100 - percent_complete
				time_remaining = (percentage_left/percent_complete)*time_training

				days1 = int(time_remaining // 86400)
				hours1 = int(time_remaining % 86400 // 3600)
				minutes1 = int((time_remaining % 3600) // 60)
				seconds1 = int(time_remaining % 60)

				print(f"run {percent_complete:.2f}%", "Epoch:", epoch, 
					  f" ETA: {days1}d {hours1}h {minutes1}m {seconds1}s")
			total_batches_seen += 1

		accuracy_cont = correct /total_samples
		accuracy01 = correct01/total_samples
		average_loss = running_loss /len(train_dataloader)
		epoch_acc.append(accuracy_cont* 100)
		counter.append(epoch_number)
		epoch_number += 1
		loss_history.append(average_loss)

		m.tb.add_scalar("Accuracy",accuracy_cont * 100, epoch)
		m.tb.add_scalar("Accuracy for binary labels", accuracy01 * 100, epoch)
		m.tb.add_scalar("Loss", average_loss, epoch)
		m.end_epoch()
	return net, counter, loss_history, epoch_acc
best_validation_accuracy = 0.0
best_model = None
best_optimizer = None
m = RunManager()
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()

start_whole_run = time.time()
total_parameters = len(b)
total_parameters_trained = 0
Epoch_num = []
ETA_list = []
Accuracy_list = []
Loss_list = []
learning_rate_list = []
Batch_size_list = []
days1 = 0
hours1 = 0
minutes1 = 0
seconds1 = 0
end_parameter_set = 0
best_validation_loss = 100

for run in b:
	torch.cuda.empty_cache()
	total_parameters_trained += 1
	print(f"Running:{run}")
	train_dataloader = torch.utils.data.DataLoader(siamese_dataset,batch_size = run.batch_size, shuffle=True)
	model = SiameseNetwork()
	model = model.cuda()
	optimizer = run.op(model.parameters(), lr=run.lr)
	pre_train_time = time.time()
	model, counter, loss_history, epoch_acc = train(model, train_dataloader, criterion, optimizer, device = torch.device("cuda:0"))
	post_train_time = time.time()
	total_time = post_train_time - pre_train_time
	#accuracy, avg_loss = m.validate(model, validation_dataloader)
	Epoch_num.append(run.number_epochs)
	ETA_list.append(total_time)
	Accuracy_list.append(epoch_acc[-1])
	Loss_list.append(loss_history[-1])
	Batch_size_list.append(run.batch_size)
	learning_rate_list.append(run.lr)

	accuracy = epoch_acc[-1]
	loss = loss_history[-1]
	print(f"{accuracy:.2f}")
	print(f"{loss:.2f}")

	#if accuracy > best_validation_accuracy:
		#best_validation_accuracy = accuracy
		#best_model = model
		#best_optimizer = optimizer
		#filename = f"BestRun_lr{run.lr}_bs{run.batch_size}_epoch{run.number_epochs}_op{run.op}_acc{(best_validation_accuracy):.0f}_loss{(loss):.2f}.pth"
		#m.save_checkpoint(model, optimizer, run.number_epochs, filename = filename)
		#print("New best!!! ", f"Accuracy:{best_validation_accuracy:.3f}", "Loss:", f"{loss:.3f}")
	if loss < best_validation_loss:
		best_validation_loss = loss
		best_model = model
		best_optimizer1 = optimizer
		filename = f"BestRun_lr{run.lr}_bs{run.batch_size}_epoch{run.number_epochs}_op{run.op}_acc{accuracy:.1f}loss{best_validation_loss:.3f}.pth"
		m.save_checkpoint(model, optimizer, run.number_epochs, filename = filename)
		print(f"New best!!! Accuracy: {(accuracy):.2f}, Loss:", f"{(best_validation_loss):.3f}")

	end_parameter_set = time.time()
	m.end_run()

headers = ["Epoch Number", "ETA", "Accuracy", "Loss","Learning Rate", "Batch Size"]
csv_path = "Optimisation data.csv"
rows = zip(Epoch_num, ETA_list, Accuracy_list, Loss_list, learning_rate_list, Batch_size_list)
with open(csv_path,"w", newline="") as file:
	writer = csv.writer(file)
	writer.writerow(headers)
	for row in rows:
		writer.writerow(row)
print("Optimisation complete")