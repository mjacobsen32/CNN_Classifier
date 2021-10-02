from .run import run
from helper_functions.get_random_subset import get_random_subset
from helper_functions.split import split
from args import Args

DS_LENGTH = 1000
TRAIN_RATIO = 0.85


def main():
    a1 = Args()

    a1.desired_length = 1000
    a1.subset_indices = get_random_subset(a1.desired_length, DS_LENGTH)
    a1.train_indices, a1.test_indices = split(a1.subset_indices, TRAIN_RATIO)
    a1.train_length = len(a1.train_indices)
    a1.test_length = len(a1.test_indices)
    a1.batch_size = "32"
    a1.test_batch_size = "32"
    a1.epochs = "50"
    a1.lr = "0.00001"
    a1.gamma = "0.7"
    a1.log_interval = "100"
    a1.num_classes = "2"
    a1.optimizer = "Adam"
    a1.model = "AlexNet"
    a1.percent_data = "1"
    a1.loss = "CEL"
    a1.input_dimension = "224"
    a1.lr_sched = "ReduceLROnPlateau"
    a1.save_model = "save_model"
    a1.no_cuda = False
    a1.dry_run = False
    a1.seed = 1
    a1.augmentations = []

    run(a1)


if __name__ == '__main__':
    main()
