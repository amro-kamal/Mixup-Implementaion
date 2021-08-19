
import torchvision
import torchvision.transforms as transforms

def cifar10_dataset(batch_size):

    train_transforms = transforms.Compose([
                                        #  transforms.RandomResizedCrop(32) ,
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor() ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                        transforms.RandomHorizontalFlip()
    ])

    val_transforms = transforms.Compose([
                                        #  transforms.Resize(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10('data/train',train=True,download=True, transform=train_transforms)
    val_data = torchvision.datasets.CIFAR10('data/val',train=False,download=True, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True,num_workers=4)
    return train_loader, val_loader