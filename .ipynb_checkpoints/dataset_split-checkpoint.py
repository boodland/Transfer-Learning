from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import FloatTensor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DatasetSplit():
    def __init__(self, path):
        self.path = path
        self.__resize_t = Resize([224, 224])
    
    def get_dataloaders(self):
        to_tensor_t = ToTensor()
        normalizer_t = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        compose_t = Compose([self.__resize_t, to_tensor_t, normalizer_t])
        to_float_t = lambda target: FloatTensor([target])
        
        train_dataset = ImageFolder(root=f'{self.path}train/', transform=compose_t, target_transform=to_float_t)
        self.train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=True)

        val_dataset = ImageFolder(root=f'{self.path}val/', transform=compose_t, target_transform=to_float_t)
        self.val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=True)
        
        test_dataset = ImageFolder(root=f'{self.path}test/', transform=compose_t, target_transform=to_float_t)
        self.test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=True)
        
        return (self.train_dataloader, self.val_dataloader, self.test_dataloader)
    
    def display_distribution(self):
        split_names = ('Train', 'Val', 'Test')
        position = np.arange(len(split_names))
        total_train = len(self.train_dataloader.dataset)
        total_val = len(self.val_dataloader.dataset)
        total_test = len(self.test_dataloader.dataset)
        split_values = [total_train/2,total_val/2,total_test/2]
        total = total_train+total_val+total_test
        bar_width = 0.22
        _, ax = plt.subplots(figsize=(10,5))
        dogs_bar = ax.bar(position - 0.6*bar_width, split_values, bar_width, color='b', align='center')
        cats_bar = ax.bar(position + 0.6*bar_width, split_values, bar_width, color='r', align='center')

        for rect in dogs_bar + cats_bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, int(height), ha='center', va='bottom', fontsize=12)

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(self.train_dataloader.dataset.classes)
        plt.yticks([])
        plt.ylabel('Number of Images', fontsize=18)
        plt.xticks(position, split_names, fontsize=18)
        plt.title(f'Dataset Split Distribution\nTotal={total}', fontsize=20)
        plt.show()
        
    def display_sample(self):
        class1 = [image for image, _ in self.train_dataloader.dataset.imgs[:10]]
        class2 = [image for image, _ in self.train_dataloader.dataset.imgs[-10:]]
        images = class1+class2
        fig, axs = plt.subplots(2, 10, figsize=(50, 10))
        fig.suptitle('Dataset Sample', fontsize=60)
        flat_axs = axs.reshape(-1)
        for ax_ind, ax in enumerate(flat_axs):
            image = self.__resize_t(Image.open(images[ax_ind]))
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.imshow(image)
        plt.show()