from model.args import parser
from torchvision import transforms
from model.plotting import Plots
import logging
import os, shutil
import pandas as pd

def main():
    args = parser.parse_args()
    c = args.constants

    if c == 'scatterPlotConstants':
        from custom_datasets.threeLines import ThreeLinesScatterPlot as Dataset
        import constants.scatterPlotConstants as constants

    try:
        p = os.path.join(constants.output_folder, args.output_folder)
        constants.output_folder = p
        os.mkdir(p)
    except:
        if args.output_folder == 'test':
            print("REMOVING TEST DIR")
            new_folder=args.output_folder
        else:
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

    logger = logging.getLogger(' create_model.py ')
    constants.LEVEL = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(constants.LEVEL)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    constants.LOG_FILE = os.path.join(constants.output_folder, args.output_folder+'.log')
    file_handler = logging.FileHandler(constants.LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("BEGIN")
    
    logger.info("output folder: {}".format(args.output_folder))
    logger.info("constants: {}".format(args.constants))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("optimization function: {}".format(args.optim))
    logger.info("loss function: {}".format(args.loss))
    logger.info("learning rate: {}".format(args.learning_rate))
    logger.info("learning rate scheduler: {}".format(args.scheduler))
    logger.info("step size: {}".format(args.step_size))
    logger.info("gamma: {}".format(args.gamma))
    logger.info("epochs: {}".format(args.epochs))
    logger.info("model: {}".format(args.model))
    logger.info("device: {}".format(args.device))
    logger.info("save model: {}".format(args.save_model))
    
    dataset = Dataset(constants.csv_path, constants.images_path, tf, None)
    logger.debug
    logger.info("img_dir: {}".format(dataset.img_dir))
    logger.debug("img_labels: {}".format(dataset.img_labels))
    
    from model.new_model import Model
    
    model = Model(args.device, args.model, constants.num_classes, dataset)
    logger.info("model: {}".format(model.model))
    model.set_loss_func(args.loss)
    model.set_optimization_func(args.optim, args.learning_rate)
    model.set_lr_sched(args.scheduler, args.gamma, args.step_size)
    model.set_subset_indices(train=0.7, validation=0.15, test=0.15)
    
    logger.debug("train_set: {}".format(model.train_set.__dict__))
    logger.debug("val_set: {}".format(model.val_set.__dict__))
    logger.debug("test_set: {}".format(model.test_set.__dict__))
    
    model.set_subsets(modelLoaderKwargs)
    
    '''
        Formatting data loaders into dataframes are extremely time consuming.
        Uncomment if a sanity check for operational data loaders is required.
    '''
    #logger.debug("train_loader: {}".format(pd.DataFrame(model.train_loader, columns=['image', 'class'])))
    #logger.debug("validation_loader: {}".format(pd.DataFrame(model.val_loader, columns=['image', 'class'])))
    #logger.debug("test_loader: {}".format(pd.DataFrame(model.test_loader, columns=['image', 'class'])))
    model.train(args.epochs)
    model.test()
    matrix, accuracy, multi = model.results(os.path.join(constants.output_folder, args.output_folder))
    logger.info(matrix)
    logger.info(accuracy)
    logger.info(multi)

    Plots.plot_val_acc_loss(False,
                            model.loss_list, 
                            model.validation_accuracy_list, 
                            os.path.join(constants.output_folder, 
                                            args.output_folder+'.png'))

    
    if args.save_model:
        model.save_model(constants.output_folder)



if __name__ == '__main__':
    main()