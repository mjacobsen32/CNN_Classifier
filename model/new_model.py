from unittest import loader
import torch
import torchvision
from .base_model import BaseModel
from math import floor
import time
import sklearn.metrics as metrics


class Model(BaseModel):
    def __init__(self, device_str, model_name, num_classes, ds):
        BaseModel.__init__(self, device_str, model_name, num_classes, ds)
        self.weights = []
        self.loss_func = None
        self.lr_sched = None
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_accuracy_list = []
        self.validation_accuracy_list = []
        self.loss_list = []


    def save_model(self, path_to_save):
        torch.save(self.model.state_dict(), path_to_save)

    def train_epoch(self, epoch, total):
        self.model.train()
        loss = 0.0
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(features)
            loss = self.loss_func(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_list.append(loss)

            if not batch_idx % 50:
                print('Epoch: %d/%d | Batch: %4d/%4d | Loss: %.4f\n'%(
                    epoch, total, batch_idx, len(self.train_loader), loss))

    def predictions(self, loader):
        y_pred = []
        y_true = []
        for features, targets in loader:
            features, targets = features.to(self.device), targets.to(self.device)
            _, predicted_labels = torch.max(self.model(features), 1)
            y_pred.append(predicted_labels)
            y_true.append(targets)
        return y_true, y_pred
    

    def compute_accuracy(self, loader):
        y_true, y_pred = self.predictions(loader=loader)
        return metrics.balanced_accuracy_score(y_true=y_true, y_score=y_pred)


    def train(self, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            self.train_epoch(epoch, epochs)
            self.scheduler.step()
            self.model.eval()
            with torch.set_grad_enabled(False):
                self.train_accuracy_list.append(self.compute_accuracy(set=self.train_loader))
                self.validation_accuracy_list.append(self.compute_accuracy(set=self.val_loader))
            print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%%\n' % (
              epoch, epochs, self.train_accuracy_list[-1] , self.validation_accuracy_list[-1]))
        total_time = time.time() - start_time


    def validation(self):
        pass

    def test(self):
        self.model.eval()
        with torch.set_grad_enabled(False):
            y_true, y_pred = self.predictions(loader=self.test_loader)
            print(metrics.confusion_matrix(y_true=y_true, y_score=y_pred, labels=self.labels))
            print(metrics.balanced_accuracy_score(y_true=y_true, y_score=y_pred))
            print(metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='weighted'))
            print(metrics.precision_recall_fscore_support(y_true=y_true, y_score=y_pred, average='macro'))
        

    def set_subsets(self, kwargs):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.val_set, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, **kwargs)

    def get_weights(self):
        w = 1 / self.num_classes
        return torch.tensor([w for _ in range(0, self.num_classes)]).to(self.device) # for now all weights are the same

    def set_subset_indices(self, train, validation, test):
        train_len = floor(len(self.dataset) * train)
        test_len = floor(len(self.dataset) * test)
        val_len = len(self.dataset) - train_len - test_len
        self.train_set = torch.utils.data.Subset(self.dataset, range(0, train_len)) 
        self.val_set = torch.utils.data.Subset(self.dataset, range(train_len, train_len+val_len))
        self.test_set = torch.utils.data.Subset(self.dataset, range(train_len+val_len, len(self.dataset)))

    def set_loss_func(self, loss_str):
        if loss_str == "MSELoss":
            self.loss_func = torch.nn.MSELoss()
        elif loss_str == "CEL":
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.get_weights(),
                                            reduction='sum')
        elif loss_str == "BCE":
            self.loss_func = torch.nn.BCELoss()
        elif loss_str == "BCEL":
            self.loss_func = torch.nn.BCEWithLogitsLoss()

    def set_optimization_func(self, opt_str, lr):
        if opt_str == "AdaDelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        elif opt_str == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        elif opt_str == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr)
        elif opt_str == "RProp":
            self.optimizer = torch.optim.Rprop(self.model.parameters())
        elif opt_str == "AdaGrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters())

    def set_lr_sched(self, sched_str, gamma, step_size):
        if sched_str == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.1, patience=2, verbose=True)
        elif sched_str == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, 
                                                        gamma=gamma, verbose=False)
        
    
