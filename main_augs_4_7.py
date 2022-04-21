from helper_functions import get_class_weights, write_to_file
from run import run
from helper_functions import get_random_subset
from helper_functions import split
from helper_functions import get_class_count
from args import Args
import constants as c
import datetime
import sys
from create_images import create_images

DS_LENGTH = 100000
TRAIN_RATIO = 0.85

MODEL_LIST = ["AlexNet", "DConvNetV2", "GoogleNet", "InceptionV3", "DenseNet169", "ResNet50", "VGG13", "squeezenet1_1"]

def main():
    a1 = Args()

    a1.augmentations = "aug" # Ranges from 1-8 'aug' is no augmentations applied

    a1.batch_size = 64
    a1.test_batch_size = 64
    a1.lr = 0.0001
    a1.step_size = 10
    a1.gamma = 0.1

    a1.optimizer = "Adam"
    a1.model = MODEL_LIST[0]
    a1.loss = "CEL"
    a1.lr_sched = "StepLR"
    a1.augmentations = "none"
    #a1.cross_validation = False

    a1.train_indices = range(0, 85000, 1) # CHANGED OFTEN
    a1.validation_indices = range(85000, 100000, 1) # CHANGED OFTEN
    a1.test_indices = range(100000, 115000, 1) # CHANGED OFTEN
    c.outputs = "../results/aug_results-4-7-2022/" # CHANGED OFTEN
    a1.epochs = 25 # CHANGED OFTEN

    a1.train_length = len(a1.train_indices)
    a1.validation_length = len(a1.validation_indices)
    a1.test_length = len(a1.test_indices)
    a1.log_interval = 100
    a1.num_classes = 2
    #a1.percent_data = 100 # CHANGED OFTEN
    #a1.input_dimension = 224

    a1.save_model = True
    a1.no_cuda = False
    a1.dry_run = False
    a1.seed = 32
    
    a1.train_class_count = get_class_count(a1.train_indices, a1.num_classes)
    a1.validation_class_count = get_class_count(a1.validation_indices, a1.num_classes)
    a1.predicted_class_count = []
    a1.class_weights = get_class_weights(a1.train_class_count, 
            a1.num_classes, a1.train_length)
    a1.tb_dir = c.tb_parent_dir

    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug1'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug2'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug3'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug4'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug5'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug6'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug7'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)
    
    now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    a1.start = now
    a1.augmentations = 'aug8'
    a1.output_file_name = str(c.outputs) + a1.augmentations
    a1.images_folder = c.base_path + a1.augmentations
    create_images(a1.images_folder, a1.augmentations, c.original_path)
    run(a1)


if __name__ == '__main__':
    main()
