import torch 
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ImbalanceDataset(Dataset):
    
    def __init__(self, dataset, train_weights, test_weights):
        self.train_weights = train_weights
        self.test_weights = test_weights
        self.train = dataset.train
        #select train_index
        train_index = []
        for i, w in enumerate(self.train_weights):
          index = np.where(dataset.train_labels == i)[0]
          np.random.shuffle(index)
          train_index = np.concatenate((train_index, index[:w]))
        self.train_data = dataset.train_data[train_index]
        self.train_labels = dataset.train_labels[train_index]
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        #select test_index
        test_index = []
        for i, w in enumerate(self.test_weights):
          index = np.where(dataset.test_labels == i)[0]
          np.random.shuffle(index)
          test_index = np.concatenate((test_index, index[:w]))
        self.test_data = dataset.test_data[test_index]
        self.test_labels = dataset.test_labels[test_index]

        self.len = len(self.train_labels)

        #self.train_data = self.train_data.unsqueeze(1) #make [B,28,28]->[B,1,28,28]
        #self.test_data = self.test_data.unsqueeze(1)
        
    def __getitem__(self, index):
        if self.train:

          img, target = self.train_data[index], int(self.train_labels[index])
        else:
          img, target = self.test_data[index], int(self.test_labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
          img = self.transform(img)

        if self.target_transform is not None:
          target = self.target_transform(target)
        return img, target
       
    def __len__(self):
        return len(self.train_labels) if self.train else len(self.test_labels)
    
def get_dataset(name):
  if name == "minst":
    train_weights=np.asarray([4000,2000,1000,750,500,350,200,100,60,40])
    test_weights = np.asarray([980,1135,1032,1010,982,892,958,1028,974,1009])
    trainset = datasets.MNIST(
          root='./MNIST/',
          train=True,
          transform=transforms.ToTensor(),
          download=True
    )

    testset = datasets.MNIST(
          root='./MNIST/',
          train=False,
          transform=transforms.ToTensor(),
          download=True
    )
    trainset = ImbalanceDataset(trainset, train_weights, test_weights)
    testset = ImbalanceDataset(testset, train_weights, test_weights)
  elif name == "fminst":
    train_weights=np.asarray([4000,2000,1000,750,500,350,200,100,60,40])
    test_weights = np.asarray([1000,1000,1000,1000,1000,1000,1000,1000,1000,1000])
    trainset = datasets.FashionMNIST(
          root='./FMNIST/',
          train=True,
          transform=transforms.ToTensor(),
          download=True
    )

    testset = datasets.FashionMNIST(
          root='./FMNIST/',
          train=False,
          transform=transforms.ToTensor(),
          download=True
    )    
    trainset = ImbalanceDataset(trainset, train_weights, test_weights)
    testset = ImbalanceDataset(testset, train_weights, test_weights)
  elif name == "cifar":
    trainset = datasets.CIFAR10(
          root='./CIFAR10/',
          train=True,
          transform=transforms.ToTensor(),
          download=True
    )

    testset = datasets.CIFAR10(
          root='./CIFAR10/',
          train=False,
          transform=transforms.ToTensor(),
          download=True
    )
  elif name == "svhn":
    trainset = datasets.SVHN(
          root='./SVHN/',
          transform=transforms.ToTensor(),
          download=True
    )

    testset = datasets.SVHN(
          root='./SVHN/',
          transform=transforms.ToTensor(),
          download=True
    )    
  elif name == "celebA":
    trainset = datasets.CelebA(
          root='./CelebA/',
          train=True,
          transform=transforms.ToTensor(),
          download=True
    )

    testset = datasets.CelebA(
          root='./CelebA/',
          train=False,
          transform=transforms.ToTensor(),
          download=True
    )   
  return trainset, testset
