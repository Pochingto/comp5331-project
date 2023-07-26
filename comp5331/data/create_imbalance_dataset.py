import torch 
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset


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


        #select test_index
        test_index = []
        for i, w in enumerate(self.test_weights):
          index = np.where(dataset.test_labels == i)[0]
          np.random.shuffle(index)
          test_index = np.concatenate((test_index, index[:w]))
        self.test_data = dataset.test_data[test_index]
        self.test_labels = dataset.test_labels[test_index]

        self.len = len(self.train_labels)


    def __getitem__(self, index):
        return (self.train_data[index], self.train_labels[index]) if self.train else (self.test_data[index], self.test_labels[index])

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
