from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

class metrics:
    def confusion_matrix(self):
        print(confusion_matrix(y_true=y_true, y_score=y_pred, labels=labels))

    def accuracy(self, ):
        return balanced_accuracy_score(y_true=y_true, y_score=y_pred)

    def roc_auc(self):
        print(roc_auc_score(y_true=y_true, y_score=y_pred, average='weighted'))

    def prec_rec_f_supp(self):
        print(precision_recall_fscore_support(y_true=y_true, y_score=y_pred, average='macro'))

    def class_report(self):
        print()

    def plot_roc_auc(self):
        pass