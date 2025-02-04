import torch
import numpy as np
from tqdm import tqdm
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
              
    def run(self, num_epochs=10, points_per_epoch=10):
        self.num_epochs = num_epochs
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.test_acc = []
        
        self.total_train = len(self.train_dataloader)
        self.report_every = int(self.total_train/points_per_epoch)
        for num_epoch, _ in enumerate(range(self.num_epochs), 1):
            self.current_epoch = num_epoch
            self.stats = {}
            with tqdm(total=self.total_train) as pbar:
                self.pbar = pbar
                self.__train()
                self.__evaluate()
                self.pbar.set_description(f'Epoch {self.current_epoch}')
        self.__predict()
        
        torch.cuda.empty_cache()

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
        self.pbar.set_description('Training')
        for iter_num, batch in enumerate(self.train_dataloader, 1):
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

            if iter_num % self.report_every == 0 or iter_num == self.total_train:
                loss_mean = np.mean(epoch_loss)
                acc_mean = np.mean(epoch_acc)
                self.stats['train loss'] = loss_mean
                self.stats['train acc'] = acc_mean
                self.pbar.set_postfix(self.stats)
                
            self.pbar.update(1)
            
        self.train_loss.append(loss_mean)
        self.train_acc.append(acc_mean)
        
    def __evaluate(self):
        evaluate_loss = []
        evaluate_acc = []
        self.model.eval()
        self.pbar.set_description('Validating')
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

                loss_value = loss.item()
                acc_value = self.__metric(outputs, labels)

                evaluate_loss.append(loss_value)
                evaluate_acc.append(acc_value)

                loss_mean = np.mean(evaluate_loss)
                acc_mean = np.mean(evaluate_acc)
                self.stats['val loss'] = loss_mean
                self.stats['val acc'] = acc_mean
                self.pbar.set_postfix(self.stats)

        self.val_loss.append(loss_mean)
        self.val_acc.append(acc_mean)
                        
        torch.cuda.empty_cache()
                            
    def __predict(self):
        self.model.eval()
        total = len(self.test_dataloader)
        with torch.no_grad():
            with tqdm(enumerate(self.test_dataloader, 1), total=total, desc='Predictions') as test_pbar:
                for iter_num, batch in test_pbar:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                    outputs = self.model(inputs)

                    acc_value = self.__metric(outputs, labels)
                    self.test_acc.append(acc_value)

                    acc_mean = np.mean(self.test_acc)
                    test_pbar.set_postfix({'acc avg': acc_mean})
                    
        torch.cuda.empty_cache()
    
    def display_result(self):
        fontsize = 17
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        ax_left, ax_right = axs
        ax_left.plot(self.train_loss, label='Train')
        ax_left.plot(self.val_loss, label='Val')
        ax_left.set_title('Training Loss Cost', fontsize=fontsize)
        ax_left.set_ylabel('Loss', fontsize=fontsize)
        ax_left.set_xlabel('Epochs', fontsize=fontsize)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.legend()
        ax_left.set_ylim(bottom=0)

        ax_right.plot(self.train_acc, label='Train')
        ax_right.plot(self.val_acc, label='Val')
        ax_right.axhline(np.mean(self.test_acc), color='g', label='Test Avg')
        ax_right.set_title('Training Accuracy', fontsize=fontsize)
        ax_right.set_ylabel('Accuracy (%)', fontsize=fontsize)
        ax_right.set_xlabel('Epochs', fontsize=fontsize)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.legend()
        ax_right.set_ylim(top=105)

        xticks = np.arange(0, self.num_epochs)
        xlabels = range(1, self.num_epochs+1)
        ax_left.set_xticks(xticks)
        ax_left.set_xticklabels(xlabels)
        ax_right.set_xticks(xticks)
        ax_right.set_xticklabels(xlabels)

        plt.show()
        
    def save_result(self, filename):
        train_result = (self.train_loss, self.train_acc)
        val_result = (self.val_loss, self.val_acc)
        results = (train_result, val_result, self.test_acc, self.num_epochs)
        torch.save(results, filename)
        
    @staticmethod
    def display_results(traditional_filename, transfer_filename):
        traditional_data = torch.load(traditional_filename)
        traditional_train, traditional_val, traditional_test, traditional_num_epochs = traditional_data
        transfer_data = torch.load(transfer_filename)
        transfer_train, transfer_val, transfer_test, transfer_num_epochs = transfer_data
        
        fontsize = 17
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        ax_left, ax_right = axs
        ax_left.plot(traditional_train[0], label='Train Traditional')
        ax_left.plot(traditional_val[0], label='Val Traditional')
        ax_left.plot(transfer_train[0], label='Train Transfer')
        ax_left.plot(transfer_val[0], label='Val Transfer')
        ax_left.set_title('Training Loss Cost', fontsize=fontsize)
        ax_left.set_ylabel('Loss', fontsize=fontsize)
        ax_left.set_xlabel('Epochs', fontsize=fontsize)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.legend()
        ax_left.set_ylim(bottom=0)

        ax_right.plot(traditional_train[1], label='Train Traditional')
        ax_right.plot(traditional_val[1], label='Val Traditional')
        ax_right.axhline(np.mean(traditional_test), color='m', label='Test Avg Traditional')
        ax_right.plot(transfer_train[1], label='Train Transfer')
        ax_right.plot(transfer_val[1], label='Val Transfer')
        ax_right.axhline(np.mean(transfer_test), color='c', label='Test Avg Transfer')
        ax_right.set_title('Training Accuracy', fontsize=fontsize)
        ax_right.set_ylabel('Accuracy (%)', fontsize=fontsize)
        ax_right.set_xlabel('Epochs', fontsize=fontsize)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.legend()
        ax_right.set_ylim(top=104)

        num_epochs = np.max([traditional_num_epochs, transfer_num_epochs])
        xticks = np.arange(0, num_epochs)
        xlabels = range(1, num_epochs+1)
        ax_left.set_xticks(xticks)
        ax_left.set_xticklabels(xlabels)
        ax_right.set_xticks(xticks)
        ax_right.set_xticklabels(xlabels)

        plt.show()
        
        