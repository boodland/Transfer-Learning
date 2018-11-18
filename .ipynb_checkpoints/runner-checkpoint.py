import torch
import numpy as np
from tqdm import tnrange, tqdm_notebook
import matplotlib.pyplot as plt

class Runner():
    def __init__(self, dataloaders, model, loss, optimizer):
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.__device)
              
    def run(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.test_acc = []
        
        for num_epoch, _ in tqdm_notebook(enumerate(range(self.num_epochs), 1), total=self.num_epochs, desc='Epochs'):
            self.current_epoch = num_epoch
            self.__train()
            self.__evaluate()
        self.__predict()

    def __metric(self, outputs, labels):
        predicted = (outputs >= 0.5).float()
        total = len(labels)
        correct = (predicted == labels).sum()
        acc_value = 100 * float(correct) / total
        return acc_value
        
    def __train(self):
        epoch_loss = []
        epoch_acc = []
        self.model.train()
        total_train = len(self.train_dataloader)
        total_val = len(self.val_dataloader)
        log_each = (total_train/total_val)
        with tqdm_notebook(
            enumerate(self.train_dataloader, 1), total=total_train, desc=f'Training {self.current_epoch}'
        ) as train_pbar:
            for iter_num, batch in train_pbar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_value = loss.item()
                acc_value = self.__metric(outputs, labels)
                
                epoch_loss.append(loss_value)
                epoch_acc.append(acc_value)
                
                if int(iter_num % log_each) == 0:
                    loss_mean = np.mean(epoch_loss)
                    acc_mean = np.mean(epoch_acc)
                    
                    self.train_loss.append(loss_mean)
                    self.train_acc.append(acc_mean)
                    
                    train_pbar.set_postfix({'loss avg': loss_mean, 'acc avg': acc_mean})
                    
        torch.cuda.empty_cache()
        
    def __evaluate(self):
        epoch_loss = []
        epoch_acc = []
        self.model.eval()
        total = len(self.val_dataloader)
        with torch.no_grad():
            with tqdm_notebook(enumerate(self.val_dataloader, 1), total=total, desc=f'Validating {self.current_epoch}') as val_pbar:
                for iter_num, batch in val_pbar:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels)
                    
                    loss_value = loss.item()
                    acc_value = self.__metric(outputs, labels)
                    
                    epoch_loss.append(loss_value)
                    epoch_acc.append(acc_value)
                    
                    loss_mean = np.mean(epoch_loss)
                    acc_mean = np.mean(epoch_acc)
                    
                    self.val_loss.append(loss_mean)
                    self.val_acc.append(acc_mean)

                    val_pbar.set_postfix({'loss avg': loss_mean, 'acc avg': acc_mean})
                    
        torch.cuda.empty_cache()
                    
    def __predict(self):
        self.model.eval()
        total = len(self.test_dataloader)
        with torch.no_grad():
            with tqdm_notebook(enumerate(self.test_dataloader, 1), total=total, desc='Predicting') as test_pbar:
                for iter_num, batch in test_pbar:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                    outputs = self.model(inputs)

                    acc_value = self.__metric(outputs, labels)
                    self.test_acc.append(acc_value)

                    acc_mean = np.mean(self.test_acc)
                    test_pbar.set_postfix({'acc avg': acc_mean})
                    
        torch.cuda.empty_cache()
        
    @staticmethod
    def __smooth_curve(points, factor=0.8):
        smoothed_points = [points[1]]
        for point in points[1:]:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        return smoothed_points
    
    def display_result(self):
        fontsize = 17
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        ax_left, ax_right = axs
        ax_left.plot(self.__smooth_curve(self.train_loss), label='Train')
        ax_left.plot(self.__smooth_curve(self.val_loss), label='Val')
        ax_left.set_title('Training Loss Cost', fontsize=fontsize)
        ax_left.set_ylabel('Loss', fontsize=fontsize)
        ax_left.set_xlabel('Epochs', fontsize=fontsize)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.legend()
        ax_left.set_ylim(bottom=0)

        ax_right.plot(self.__smooth_curve(self.train_acc), label='Train')
        ax_right.plot(self.__smooth_curve(self.val_acc), label='Val')
        ax_right.axhline(np.mean(self.test_acc), color='g', label='Test Avg')
        ax_right.set_title('Training Accuracy', fontsize=fontsize)
        ax_right.set_ylabel('Accuracy (%)', fontsize=fontsize)
        ax_right.set_xlabel('Epochs', fontsize=fontsize)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.legend()
        ax_right.set_ylim(top=105)

        xticks = np.arange(0, len(self.train_acc)+1, len(self.train_acc)/self.num_epochs)
        xlabels = range(self.num_epochs+1)
        ax_left.set_xticks(xticks)
        ax_left.set_xticklabels(xlabels)
        ax_right.set_xticks(xticks)
        ax_right.set_xticklabels(xlabels)

        plt.show()
        
    def save_result(self, filename):
        train_result = (self.train_loss, self.train_acc)
        val_result = (self.val_loss, self.val_acc)
        results = (train_result, val_result, self.test_acc)
        torch.save(results, filename)
        
    @staticmethod
    def display_results(traditional_filename, transfer_filename):
        traditional_data = torch.load(traditional_filename)
        traditional_train, traditional_val, traditional_test = traditional_data
        transfer_data = torch.load(transfer_filename)
        transfer_train, transfer_val, transfer_test = transfer_data
        
        fontsize = 17
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        ax_left, ax_right = axs
        ax_left.plot(Runner.__smooth_curve(traditional_train[0]), label='Train Traditional')
        ax_left.plot(Runner.__smooth_curve(traditional_val[0]), label='Val Traditional')
        ax_left.plot(Runner.__smooth_curve(transfer_train[0]), label='Train Transfer')
        ax_left.plot(Runner.__smooth_curve(transfer_val[0]), label='Val Transfer')
        ax_left.set_title('Training Loss Cost', fontsize=fontsize)
        ax_left.set_ylabel('Loss', fontsize=fontsize)
        ax_left.set_xlabel('Epochs', fontsize=fontsize)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.legend()
        ax_left.set_ylim(bottom=0)

        ax_right.plot(Runner.__smooth_curve(traditional_train[1]), label='Train Traditional')
        ax_right.plot(Runner.__smooth_curve(traditional_val[1]), label='Val Traditional')
        ax_right.axhline(np.mean(traditional_test), color='m', label='Test Avg Traditional')
        ax_right.plot(Runner.__smooth_curve(transfer_train[1]), label='Train Transfer')
        ax_right.plot(Runner.__smooth_curve(transfer_val[1]), label='Val Transfer')
        ax_right.axhline(np.mean(transfer_test), color='c', label='Test Avg Transfer')
        ax_right.set_title('Training Accuracy', fontsize=fontsize)
        ax_right.set_ylabel('Accuracy (%)', fontsize=fontsize)
        ax_right.set_xlabel('Epochs', fontsize=fontsize)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.legend()
        ax_right.set_ylim(top=104)

        num_epochs = 10
        xticks = np.arange(0, len(traditional_train[1])+1, len(traditional_train[1])/num_epochs)
        xlabels = range(num_epochs+1)
        ax_left.set_xticks(xticks)
        ax_left.set_xticklabels(xlabels)
        ax_right.set_xticks(xticks)
        ax_right.set_xticklabels(xlabels)

        plt.show()
        
        