import formatter
from model.new_model import Model
from model.args import parser
from custom_datasets import phytoplankton, threeLines
from constants import scatterPlotConstants, planktonConstants
from torchvision import transforms
from model.plotting import Plots
import logging
import os, shutil
import pandas as pd

LEVEL = logging.DEBUG

def main():
    args = parser.parse_args()
    c = args.constants

    if c == 'scatterPlotConstants':
        from custom_datasets.threeLines import ThreeLinesScatterPlot as Dataset
        import constants.scatterPlotConstants as constants
    elif c == 'planktonConstants':
        from custom_datasets.phytoplankton import PhytoplanktonDataset as Dataset
        import constants.planktonConstants as constants

    try:
        p = os.path.join(constants.output_folder, args.output_folder)
        os.mkdir(p)
    except:
        print("{} already exists!\nRetype '{}' to overwrite, press enter to exit".format(p, args.output_folder))
        new_folder = input()
        if new_folder == args.output_folder:
            shutil.rmtree(p)
            os.mkdir(p)
        else:
            return

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225])
    ])

    modelLoaderKwargs = {'batch_size': args.batch_size, 'shuffle': True}
    if args.device == 'cuda':
        modelLoaderKwargs.update({'num_workers': 0,
                                    'pin_memory': True,
                                    'shuffle': True})

    logger = logging.getLogger('__name__')
    logger.setLevel(LEVEL)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler(os.path.join(constants.output_folder, args.output_folder, args.output_folder+'.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug("BEGIN")

    dataset = Dataset(constants.csv_path, constants.images_path, tf, None)
    logger.debug("img_dir: {}".format(dataset.img_dir))
    logger.debug("img_labels: {}".format(dataset.img_labels))
    
    model = Model(args.device, args.model, constants.num_classes, dataset)
    logger.debug("model: {}".format(model.model))
    model.set_loss_func(args.loss)
    model.set_optimization_func(args.optim, args.learning_rate)
    model.set_lr_sched(args.scheduler, args.gamma, args.step_size)
    model.set_subset_indices(train=0.7, validation=0.15, test=0.15)
    
    logger.debug("train_set: {}".format(model.train_set.__dict__))
    logger.debug("val_set: {}".format(model.val_set.__dict__))
    logger.debug("test_set: {}".format(model.test_set.__dict__))
    
    model.set_subsets(modelLoaderKwargs)
    logger.debug("train_loader: {}".format(pd.DataFrame(model.train_loader, columns=['image', 'class'])))
    logger.debug("validation_loader: {}".format(pd.DataFrame(model.val_loader, columns=['image', 'class'])))
    logger.debug("test_loader: {}".format(pd.DataFrame(model.test_loader, columns=['image', 'class'])))
    breakpoint()
    model.write_parameters(os.path.join(constants.output_folder, args.output_folder),
                            args.model, args.loss, args.optim, args.scheduler)
    model.train(args.epochs)
    model.test()
    model.results(os.path.join(constants.output_folder, args.output_folder))

    Plots.plot_val_acc_loss(False,
                            model.loss_list, 
                            model.validation_accuracy_list, 
                            os.path.join(constants.output_folder, 
                                            args.output_folder, 
                                            args.output_folder+'.png'))

    
    if args.save_model:
        model.save_model(args.output_folder)



if __name__ == '__main__':
    main()