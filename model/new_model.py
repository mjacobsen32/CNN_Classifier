import torch
import torchvision
from .base_model import BaseModel
from math import floor
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score


class Model(BaseModel):
    def __init__(self, device_str, model_name, num_classes, ds):
        BaseModel.__init__(self, device_str, model_name, num_classes, ds)
        self.weights = []
        self.loss_func = None
        self.optim_func = None
        self.lr_sched = None
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss_list = []
        self.train_accuracy_list = []
        self.validation_accuracy_list = []
        self.labels = ['one', 'two', 'three']


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
            self.optim_func.zero_grad()
            loss.backward()
            self.optim_func.step()
            self.loss_list.append(loss)

            if not batch_idx % 50:
                print('Epoch: %d/%d | Batch: %4d/%4d | Loss: %.4f\n'%(
                    epoch, total, batch_idx, len(self.train_loader), loss))


    def confusion_matrix(self):
        print(confusion_matrix(y_true, y_pred, self.labels))

    def accuracy(self):
        balanced_accuracy_score(y_true, y_pred)

    def roc_auc(self):
        print(roc_auc_score(y_true=y_true, y_score=y_score, average='weighted'))

    def prec_rec_f_supp(self):
        print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

    def class_report(self):
        print()

    def plot_roc_auc(self):
        pass

    def compute_accuracy(self, test=False,)


    def train(self, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            self.train_epoch(epoch, total)
            self.train_accuracy_list.append(self.compute_accuracy(set=self.train_loader))
            self.validation_accuracy_list.append(self.compute_accuracy(set=self.val_loader))


    def validation(self):
        pass

    def test(self):
        pass

    def subsets(self, kwargs):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_set, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, **kwargs)

    def weights(self):
        w = 1 / self.num_classes
        return [w for _ in range(0, self.num_classes)] # for now all weights are the same

    def subset_inds(self, train, _, test):
        train_len = floor(len(self.dataset) * train)
        test_len = floor(len(self.dataset) * test)
        val_len = len(self.dataset) - self.train_len - self.test_len
        self.train_set = torch.utils.data.Subset(self.dataset, range(0, train_len)) 
        self.val_set = torch.utils.data.Subset(self.dataset, range(train_len, train_len+val_len))
        self.test_set = torch.utils.data.Subset(self.dataset, range(train_len+val_len, len(self.dataset)))

    def loss_func(self, loss_str):
        if loss_str == "MSELoss":
            self.loss_func =torch.nn.MSELoss()
        elif loss_str == "CEL":
            self.loss_func =torch.nn.CrossEntropyLoss(weight=self.get_weights(),
                                            reduction='sum')
        elif loss_str == "BCE":
            self.loss_func =torch.nn.BCELoss()
        elif loss_str == "BCEL":
            self.loss_func =torch.nn.BCEWithLogitsLoss()

    def optimization_func(self, opt_str, lr):
        if opt_str == "AdaDelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        elif opt_str == "Adam":
            self.optimizertorch.optim.Adam(self.model.parameters(), lr)
        elif opt_str == "SGD":
            self.optimizertorch.optim.SGD(self.model.parameters(), lr)
        elif opt_str == "RProp":
            self.optimizertorch.optim.Rprop(self.model.parameters())
        elif opt_str == "AdaGrad":
            self.optimizertorch.optim.Adagrad(self.model.parameters())

    def lr_sched(self, sched_str, gamma, step_size):
        if sched_str == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.1, patience=2, verbose=True)
        elif sched_str == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, 
                                                        gamma=gamma, verbose=False)
        
    
