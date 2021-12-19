from helper_functions import get_class_weights, write_to_file
from run import run
from helper_functions import get_random_subset
from helper_functions import split
from helper_functions import get_class_count
from args import Args
import constants as c
import datetime

DS_LENGTH = 100000
TRAIN_RATIO = 0.85

MODEL_LIST = ["AlexNet", "DConvNetV2", "GoogleLeNet"]

def main():
    a1 = Args()

    a1.augmentations = "aug" # Ranges from 1-8 'aug' is no augmentations applied

    a1.desired_length = 100000
    a1.batch_size = 64
    a1.test_batch_size = 64
    a1.epochs = 20
    a1.lr = 0.0001 # Adam got to 82% accuracy quickly with 0.0001 as starting value
    a1.gamma = 1.0

    a1.optimizer = "Adam"
    a1.model = MODEL_LIST[0]
    a1.loss = "CEL"
    a1.lr_sched = "StepLR"
    a1.augmentations = "aug"

    a1.subset_indices = get_random_subset(a1.desired_length, DS_LENGTH)
    a1.train_indices, a1.test_indices = split(a1.subset_indices, TRAIN_RATIO)
    a1.train_length = len(a1.train_indices)
    a1.test_length = len(a1.test_indices)
    a1.log_interval = 100
    a1.num_classes = 2
    a1.percent_data = 100
    a1.input_dimension = 224

    a1.save_model = False
    a1.no_cuda = False
    a1.dry_run = False
    a1.seed = 32
    a1.train_class_count = get_class_count(a1.train_indices, a1.num_classes)
    a1.test_class_count = get_class_count(a1.test_indices, a1.num_classes)
    a1.predicted_class_count = []
    a1.class_weights = get_class_weights(a1.train_class_count, 
            a1.num_classes, a1.train_length)
    a1.tb_dir = c.tb_parent_dir

    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.index_file = str(c.outputs) + now

    #output = "train indices: {}\ntest indices: {}\n".format(a1.train_indices, 
    #        a1.test_indices)
    #write_to_file(output, str(a1.index_file + "_INDEXES"))
    output = ""
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.output_file_name = str(c.outputs) + now
    a1.augmentations = "aug"
    run(a1)

if __name__ == '__main__':
    main()
