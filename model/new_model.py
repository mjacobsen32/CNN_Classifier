import torch
from model.base_model import BaseModel
from math import floor
import time
import sklearn.metrics as metrics
from model.timeFunctions import timer_wrapper
import os
from collections import Counter, OrderedDict


class Model(BaseModel):
    
    @timer_wrapper
    def __init__(self, device_str, model_name, num_classes, ds):
        BaseModel.__init__(self, device_str, model_name, num_classes, ds)
        self.weights = []
        self.loss_func = None
        self.optimizer = None
        self.scheduler = None
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_accuracy_list = []
        self.validation_accuracy_list = []
        self.loss_list = []
        self.labels = [c for c in range(num_classes)]
        self.total_time = 0.0
        self.y_true = []
        self.y_pred = []

    @timer_wrapper
    def save_model(self, path_to_save):
        torch.save(self.model.state_dict(), os.path.join(path_to_save,'model.pth'))

    @timer_wrapper
    def train_epoch(self, epoch, total):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(features)
            loss = self.loss_func(logits, targets)
            running_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if not batch_idx % 5:
                print('Epoch: %d/%d | Batch: %4d/%4d | Loss: %.4f\n'%(
                    epoch+1, total, batch_idx, len(self.train_loader), loss))
        return running_loss / len(self.train_loader)

    @timer_wrapper
    def predictions(self, loader):
        y_pred = []
        y_true = []
        for features, targets in loader:
            features, targets = features.to(self.device), targets.to(self.device)
            _, predicted_labels = torch.max(self.model(features), 1)
            y_pred += torch.Tensor.cpu(predicted_labels).tolist()
            y_true += torch.Tensor.cpu(targets).tolist()
        return y_true, y_pred
    
    @timer_wrapper
    def compute_accuracy(self, loader):
        y_true, y_pred = self.predictions(loader=loader)
        return metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)

    @timer_wrapper
    def train(self, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            loss = self.train_epoch(epoch, epochs)
            self.loss_list.append(loss)
            self.scheduler.step(metrics=loss)
            self.model.eval()
            with torch.set_grad_enabled(False):
                self.train_accuracy_list.append(self.compute_accuracy(loader=self.train_loader))
                self.validation_accuracy_list.append(self.compute_accuracy(loader=self.val_loader))
            print('Epoch: %03d/%03d | Train: %0.3f%% | Validation: %3f%%\n' % (
              epoch+1, epochs, self.train_accuracy_list[-1] , self.validation_accuracy_list[-1]))
        self.total_time = time.time() - start_time

    @timer_wrapper
    def test(self):
        self.model.eval()
        with torch.set_grad_enabled(False):
            self.y_true, self.y_pred = self.predictions(loader=self.test_loader)
    
    @timer_wrapper
    def results(self, outputFolder):
        matrix = ("Confusion Matrix: \n"+str(metrics.confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, labels=self.labels))+'\n')
        accuracy = ("Balanced Accuracy Score: "+str(metrics.balanced_accuracy_score(y_true=self.y_true, y_pred=self.y_pred))+'\n')
        #f.write(metrics.roc_auc_score(y_true=y_true, y_pred=y_pred, average='weighted'))
        p_r_f_s = ("Precision Recall, FScore, Support: "+str(metrics.precision_recall_fscore_support(y_true=self.y_true, y_pred=self.y_pred, average='macro'))+'\n')
        return matrix, accuracy, p_r_f_s

    @timer_wrapper
    def set_subsets(self, kwargs):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, **kwargs)

    @timer_wrapper
    def get_weights(self):
        ds_classes = [label for _, label in self.dataset]
        count = OrderedDict(Counter(ds_classes))
        i = 0
        weights_as_list = [(c/len(self.dataset)) for c in count.values()]
        print("class count:\n{}".format(count))
        print("weights:\n{}".format(weights_as_list))
        weights_as_cuda_tensor = torch.tensor(weights_as_list).to(self.device) 
        return weights_as_cuda_tensor

    @timer_wrapper
    def set_subset_indices(self, train, validation, test):
        train_len = floor(len(self.dataset) * train)
        test_len = floor(len(self.dataset) * test)
        val_len = floor(len(self.dataset) * validation)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(dataset=self.dataset, lengths=[train_len, val_len, test_len])
        print("train_set len: {}\nval_set len: {}\ntest_set len: {}".format(
            len(self.train_set), len(self.val_set), len(self.test_set)
        ))

    @timer_wrapper
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

    @timer_wrapper
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

    @timer_wrapper
    def set_lr_sched(self, sched_str, gamma, step_size):
        if sched_str == "ROP":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.1, patience=4, verbose=True)
        elif sched_str == "Step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, 
                                                        gamma=gamma, verbose=False)
        
    
