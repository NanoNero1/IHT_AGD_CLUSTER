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
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
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
                                             shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
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

    BATCH_SIZE = 64

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    return train_loader, test_loader
    

