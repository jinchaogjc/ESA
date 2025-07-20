import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.mnist = MNIST(root=root, train=train, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        return self.mnist[index]

if __name__ == "__main__":
    mnist_train = torchvision.datasets.MNIST(root='datasets/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(root='datasets/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor())
    

    mnist_dataset = MNISTDataset(root='datasets/mnist/', train=True, transform=torchvision.transforms.ToTensor())

    # Select a sample from the dataset
    sample = mnist_dataset[0]

    # Access the image and label of the sample
    image, label = sample

    # Plot the image
    plt.imshow(image[0, :, :], cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
