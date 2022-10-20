from cProfile import label
from matplotlib import pyplot as plt

class Plots:
    def plot_val_acc_loss(locked_range, loss_list, val_acc_list, path):
        plt.style.use('classic')
        fig, ax = plt.subplots()

        ax.plot(loss_list, color='r', label='loss')
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_ylabel('validation loss')
        ax.set_xlabel("epoch")

        ax2=ax.twinx()
        ax2.plot(val_acc_list, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylabel('validation accuracy')
        if locked_range:
            ax.set_ylim(0,50)
            ax2.set_ylim(75,100)

        plt.savefig(path)
