import os
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark = False

def get_transforms(config):
    transform_train = transforms.Compose([
        transforms.Resize(tuple(config['data']['transform_train']['resize'])),
        transforms.RandomHorizontalFlip() if config['data']['transform_train']['random_horizontal_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(config['data']['transform_train']['random_rotation']) if 'random_rotation' in config['data']['transform_train'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['transform_train']['normalize']['mean'],
                             std= config['data']['transform_train']['normalize']['std'])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(tuple(config['data']['transform_val_test']['resize'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['transform_val_test']['normalize']['mean'],
                             std= config['data']['transform_val_test']['normalize']['std'])
    ])

    return transform_train, transform_val_test


def prepare_datasets(config):
    transform_train, transform_val_test = get_transforms(config)
    full_dataset = datasets.ImageFolder(root=config['data']['root'], transform=transform_train)
    dataset_size = len(full_dataset)
    targets = full_dataset.targets

    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - config['data']['train_ratio']), random_state=config['seed'])
    for train_idx, temp_idx in sss.split(np.zeros(len(targets)), targets):
        train_dataset = Subset(full_dataset, train_idx)
        temp_dataset = Subset(full_dataset, temp_idx)

    val_size = int(config['data']['val_ratio'] / (1 - config['data']['train_ratio']) * len(temp_dataset))
    test_size = len(temp_dataset) - val_size
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(config['seed']))

    val_dataset.dataset.transform = transform_val_test
    test_dataset.dataset.transform = transform_val_test

    return train_dataset, val_dataset, test_dataset, full_dataset.classes


def get_data_loaders(train_dataset, val_dataset, test_dataset, config):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size']['train'],
        shuffle=True,
        num_workers= config['data']['num_workers'],
        pin_memory= config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size']['test'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory= config['data']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = config['data']['batch_size']['test'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader, test_loader