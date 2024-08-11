import torch
import torchvision
import torchvision.transforms as transforms


# Dimitri Imports
import time
from datasets import load_dataset
from torchvision.transforms import Lambda
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def mnist_loader(batch_size):
    # Preprocess input
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.13, ), (0.3, ))])
    # Load   
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader

def cifar_loader(batch_size):
    # Preprocess input
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # Load
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    
    ## FOR THE VALIDATION
    ## SOURCE: https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
    validation_params = True

    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    return trainloader, testloader

def imagenet_loader(batch_size):
    # If the dataset is gated/private, make sure you have run huggingface-cli login
    timeOne = time.time()
    dataset = load_dataset("imagenet-1k")


    print("yes this did gather imagenet!")

    timeTwo = time.time()

    print(timeTwo - timeOne)

    # Select the first row in the dataset
    sample = dataset['train'][0]

    # Split up the sample into two variables
    datapt, label = sample['image'], sample['label']

    print(label)
    print(datapt)


    ### SOURCE: https://medium.com/@ricodedeijn/image-classification-computer-vision-from-scratch-pt-3-9d5fbcf3c363
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            # Loads the dataset that needs to be transformed
            self.dataset = load_dataset("imagenet-1k", split=f"{dataset}")

        def __getitem__(self, idx):
            # Sample row idx from the loaded dataset
            sample = self.dataset[idx]
            
            # Split up the sample example into an image and label variable
            data, label = sample['image'], sample['label']
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize to size 256x256
                Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),  # Convert all images to RGB format
                transforms.ToTensor(),  # Transform image to Tensor object
            ])
            
            # Returns the transformed images and labels
            return transform(data), torch.tensor(label)

        def __len__(self):
            return len(self.dataset)

    # Call the class to populate variable train_set with the train data
    train_set = MyDataset('train')
    test_set = MyDataset('test')

    BATCH_SIZE = 128

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,num_workers=16)

    return train_loader, test_loader

def mnist_loader(batch_size):
    # Preprocess input
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.13, ), (0.3, ))])
    # Load   
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader

def cifar100_loader(batch_size):

    
    train_data = torchvision.datasets.CIFAR100('./data', train=True, download=True)
    
    # Stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

    # calculate the mean and std along the (0, 1) axes
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    # the the mean and std
    mean=mean.tolist()
    std=std.tolist()

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(mean,std,inplace=True)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    return trainloader, testloader
    

