# 3rd-year-project


  import torch
  
  import torch.nn as nn
  
  import torch.optim as optim
  
  import torch.nn.functional as F
  
  
  
  import torchvision
  
  import torchvision.transforms as transforms
  
  
  
  import numpy as np
  
  #import pandas as pd #This line gives an error stating "No module named 'pandas'
  
  #does this mean I have to go back to conda and download pandas aswell as matplotlib from there? I cannot find any pandas link like I found for matplotlib and pytorch however I don't need it just yet
  
  import matplotlib.pyplot as plt
  
  
  #LLoading the data from the web
  
  train_set = torchvision.datasets.FashionMNIST(root='./dataFashionMNIST' #where the data is located in the directory(hopefully in my python folder on my NVME so my computer's windows does not crash on the other drive)
                                                , train=True #we want to use this data to train on
                                                , download=True #data should be downloaded if not already
                                                , transform=transforms.Compose([transforms.ToTensor()])) 
                                                
                                                #define transform tha should be used on the data elements(Turning our data to tensors)
  
  
  
  #Taking the data and doing stuff to it?
  
  train_loader = torch.utils.data.DataLoader(train_set
                                             ,batch_size = 1000)
                                             #, shuffle = True)
  
  
  
  torch.set_printoptions(linewidth=120)#sets the linewidth for pytorch output that is printed to the console????
  
  print(len(train_set))#how many images are in the training set
  
  print(train_set.train_labels)#these print the labels assigned to each of the images 0-9
  
  print(train_set.train_labels.bincount()) #count freq of occorences of each bin. We can see the split between all the image classes are equal as they have 6000 each? It is therefore balanced
  
  #If we have an unbalanced dataset the best way to balance it is to duplicate the images of the lesser label until it is classed as balanced
  
  
  
  
  #Accessing a data sample from the train set
  
  sample = next(iter(train_set)) #extract the single sample
  
  print(len(sample)) #length of the sample, this gives 2 as there are image-lable pairs
  
  print(type(sample)) #same as above explanation? part of the same explanation
  
  image, label = sample # sequence unpacking to get each items. This is the same as writing 'image =  sample[0]
  
  #                                                                                          label = sample[1]' soemntimes called deconstructing
  
  
  
  plt.imshow(image.squeeze(), cmap='gray')#plotted the image
  
  print('label:', label)#printing the label
  
  
  batch=  next(iter(train_loader))#loading a batch of images
  
  images, labels = batch#seperating the labels and images
  
  #the batch is 10 as stated previously
  
  
  grid = torchvision.utils.make_grid(images, nrow=100) #made
  the grid of all the images, nrow is the number of images per row
  
  
  plt.figure(figsize=(15,15))#
  
  plt.imshow(np.transpose(grid, (1, 2, 0)))#set plot configurations and transpose grid
  
  print('label:', labels) #check data for labels is correct
  
  
  class Lizard:#announces the class and the name of the class
  
      def __init__(self, name):#when a new instance of a class is created, have 2 parameters called self and name
          self.name = name
      #Ok I am now lost from here on out I will definitely need your help tommorrow
      def set_name(self, name):
          self.name = name
  
  
  lizard = Lizard('deep')
  
  print(lizard.name, 'first name of lizard')
  
  
  lizard.set_name('lizard')
  
  print(lizard.name, '2nd name of lizard')
  
  
  
  class Network(nn.Module):
  
      def __init__(self):
          super(Network, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=1,out_channels= 6, kernel_size=5)#from here
          self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)#these 2 are convolutional layers
          
          self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
          self.fc2 = nn.Linear(in_features=120, out_features=60)#this bit shows all the linear layers
          self.out = nn.Linear(in_features=60, out_features=10)#to here #we call these "layers" attributes. Each layer has 2 components
      
      def forward(self, t):
          #implement the forward pass
          return t
  #Apparently there is a bug in the code here on deeplizard, this works out well enough and does not break so I don't care really#
  
  
  network = Network()
  network
  
  
  #CNN Layers Deep neural network architecture
  
  class Network(nn.Module):
      
      def __init__(self):
          super(Network, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=1,out_channels= 6, kernel_size=5)#in_channel = no. colorur channels in input image, kernel_size = hyperparameter, out_channel = hyperparameter
          self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)#in_channel = no. out_channels in previous layer, kernel_size = hyperparameter, ouut_channel = hyperparameter(higher than prev. conv layer)
  
          self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)#in_features = length of flattened output from previous layer, out_features = hyperparameter
          self.fc2 = nn.Linear(in_features=120, out_features=60)#in_features = no. outfeatures form previous layer, out_features = hyperparameter(lower than prev. lin layer)
          self.out = nn.Linear(in_features=60, out_features=10)#in_features = no. outfeatures form previous layer, out_features = no. prediction classes
  
      def forward(self, t):
          return t
  #parameter vs argument
  #parameters are placeholders?
  #arguments are the values of the parameters. Eg: in_channels = 1
  #the parameter == in_channels & the argument == 1
  
  #therer are 2 types of parameters
  #Hyper parameters and Data dependentent hyperparameters
  #   Hyper parameters
  #       values that are chosen manually and arbitrarily mainly based off of trial and error and using values that have worked in the past
  #           eg: kernel_size, out_channel & out_features. We(they) chose these values and were not derived
  #   Data dependendant hyper parameters
  #       These are parameter's who's values depend on the data

  print(network.conv1.weight)
