import sys
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_loader(args):

    # Define transformations for ImageNet100
    imagenet100_transform_train = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.RandomCrop(224),                     # Randomly crop image to 224x224
        transforms.RandomHorizontalFlip(),              # Randomly flip the image horizontally
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
    ])
    
    imagenet100_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
    ])  

    # Define transformations for ImageNet200
    imagenet200_transform_train = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.RandomCrop(224),                     # Randomly crop image to 224x224
        transforms.RandomHorizontalFlip(),              # Randomly flip the image horizontally
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalize with Tiny ImageNet statistics
    ])
    
    imagenet200_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalize with ImageNet statistics
    ])

    # Define transformations for ImageNet1k
    imagenet1k_transform_train = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.RandomCrop(224),                     # Randomly crop image to 224x224
        transforms.RandomHorizontalFlip(),              # Randomly flip the image horizontally
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
    ])
    
    imagenet1k_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
    ])

    # Define transformations for CIFAR
    cifar_transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),                        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    cifar_transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./datasets/cifar10",
                                    train=True,
                                    download=True,
                                    transform=cifar_transform_train)
        testset = datasets.CIFAR10(root="./datasets/cifar10",
                                   train=False,
                                   download=True,
                                   transform=cifar_transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./datasets/cifar100",
                                     train=True,
                                     download=True,
                                     transform=cifar_transform_train)
        testset = datasets.CIFAR100(root="./datasets/cifar100",
                                    train=False,
                                    download=True,
                                    transform=cifar_transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "imagenet100":
        traindir = "./datasets/imagenet100/train"
        valdir = "./datasets/imagenet100/val"
        # trainset = CustomImageDataset(traindir, transform=imagenet100_transform_train)
        # testset = CustomImageDataset(valdir, transform=imagenet100_transform_test) if args.local_rank in [-1, 0] else None
        trainset = datasets.ImageFolder(traindir, transform=imagenet100_transform_train)
        testset = datasets.ImageFolder(valdir, transform=imagenet100_transform_test) if args.local_rank in [-1, 0] else None
    
    elif args.dataset == "tiny_imagenet200": # Tiny ImageNet (images are 64x64 pixels)
        traindir = "./datasets/tiny_imagenet200/train"
        valdir = "./datasets/tiny_imagenet200/val"
        trainset = datasets.ImageFolder(traindir, transform=imagenet200_transform_train)
        testset = datasets.ImageFolder(valdir, transform=imagenet200_transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "imagenet1k":
        traindir = "./datasets/imagenet1k/train"
        valdir = "./datasets/imagenet1k/val"
        # trainset = CustomImageDataset(traindir, transform=imagenet1k_transform_train)
        # testset = CustomImageDataset(valdir, transform=imagenet1k_transform_test) if args.local_rank in [-1, 0] else None
        trainset = datasets.ImageFolder(traindir, transform=imagenet1k_transform_train)
        testset = datasets.ImageFolder(valdir, transform=imagenet1k_transform_test) if args.local_rank in [-1, 0] else None
    else:
        raise NotImplementedError

    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              pin_memory=True)

    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=args.eval_batch_size) if testset is not None else None

    return train_loader, test_loader
