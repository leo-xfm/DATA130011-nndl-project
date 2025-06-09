import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(val_ratio = 0.1):
    
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2023, 0.1994, 0.2010]

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = base_transform
    )
    
    val_sz = int(len(train_dataset) * val_ratio)
    train_sz = len(train_dataset) - val_sz
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_sz, val_sz])
    
    train_dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        transforms.RandomCrop(32, padding=4)
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = base_transform
    )
    
    return train_dataset, val_dataset, test_dataset

def get_origin_dataset():
    

    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = base_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = base_transform
    )
    
    return train_dataset, test_dataset