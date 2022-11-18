from model.args import parser
from torchvision import transforms
from model.plotting import Plots
from model.logger import modelLog, formatter
import os, shutil
import logging


def main(a, num_classes, csv_name):
    args = parser.parse_args(a) if a else parser.parse_args()
    print(args.output_folder)
    c = args.constants
    

    if c == 'scatterPlotConstants':
        from custom_datasets.threeLines import ThreeLinesScatterPlot as Dataset
        import constants.scatterPlotConstants as constants
        
    constants.num_classes = num_classes if num_classes else constants.num_classes
    csv_path = os.path.join(constants.csv_path, csv_name) if csv_name else constants.csv_path
    
    try:
        output_folder = os.path.join(constants.output_folder, args.output_folder)
        os.mkdir(output_folder)
    except:
        if args.output_folder == 'test':
            print("REMOVING TEST DIR")
            new_folder=args.output_folder
        else:
            print("{} already exists!\nRetype '{}' to overwrite, press enter to exit".format(output_folder, args.output_folder))
            new_folder = input()
        if new_folder == args.output_folder:
            shutil.rmtree(output_folder)
            os.mkdir(output_folder)
        else:
            return
    print("Real time logging and results can be found in the <output_folder>.log in the <output_folder> directory")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225])
    ])

    modelLoaderKwargs = {'batch_size': args.batch_size, 'shuffle': True}
    if args.device == 'cuda':
        modelLoaderKwargs.update({'num_workers': 0,
                                    'pin_memory': True,
                                    'shuffle': True})
    
    LOG_FILE = os.path.join(output_folder, args.output_folder+'.log')
    
    file_handler = logging.FileHandler(filename=LOG_FILE, mode='w')
    file_handler.setFormatter(formatter)
    modelLog.addHandler(file_handler)
        
    modelLog.info("BEGIN")
    modelLog.info("output folder: {}".format(args.output_folder))
    modelLog.info("constants: {}".format(args.constants))
    modelLog.info("batch size: {}".format(args.batch_size))
    modelLog.info("optimization function: {}".format(args.optim))
    modelLog.info("loss function: {}".format(args.loss))
    modelLog.info("learning rate: {}".format(args.learning_rate))
    modelLog.info("learning rate scheduler: {}".format(args.scheduler))
    modelLog.info("step size: {}".format(args.step_size))
    modelLog.info("gamma: {}".format(args.gamma))
    modelLog.info("epochs: {}".format(args.epochs))
    modelLog.info("model: {}".format(args.model))
    modelLog.info("device: {}".format(args.device))
    modelLog.info("save model: {}".format(args.save_model))
    
    dataset = Dataset(csv_path, constants.images_path, tf, None)
    
    modelLog.info("img_dir: {}".format(dataset.img_dir))
    modelLog.debug("img_labels: {}".format(dataset.img_labels))
    
    from model.new_model import Model
    
    model = Model(args.device, args.model, constants.num_classes, dataset)
    modelLog.debug("model: {}".format(model.model))
    model.set_loss_func(args.loss)
    model.set_optimization_func(args.optim, args.learning_rate)
    model.set_lr_sched(args.scheduler, args.gamma, args.step_size)
    model.set_subset_indices(train=0.7, validation=0.15, test=0.15)
    
    modelLog.debug("train_set: {}".format(model.train_set.__dict__))
    modelLog.debug("val_set: {}".format(model.val_set.__dict__))
    modelLog.debug("test_set: {}".format(model.test_set.__dict__))
    
    model.set_subsets(modelLoaderKwargs)
    
    '''
        Formatting data loaders into dataframes are extremely time consuming.
        Uncomment if a sanity check for operational data loaders is required.
    '''
    #modelLog.debug("train_loader: {}".format(pd.DataFrame(model.train_loader, columns=['image', 'class'])))
    #modelLog.debug("validation_loader: {}".format(pd.DataFrame(model.val_loader, columns=['image', 'class'])))
    #modelLog.debug("test_loader: {}".format(pd.DataFrame(model.test_loader, columns=['image', 'class'])))
    
    model.train(args.epochs)
    
    model.test()
    model.results(os.path.join(output_folder, args.output_folder))

    Plots.plot_val_acc_loss(False,
                            model.loss_list, 
                            model.validation_accuracy_list, 
                            os.path.join(output_folder, 
                                            args.output_folder+'.png'))

    
    if args.save_model:
        model.save_model(output_folder)

    modelLog.removeHandler(file_handler)
    del file_handler


if __name__ == '__main__':
    
    a1 = ["-out", "SixGoogLeNet",
          "-b", "16",
          "-o", "Adam",
          "-l", "CEL",
          "-lr", "0.0001",
          "-s", "Step",
          "-ss", "5",
          "-g", "0.1",
          "-e", "2",
          "-m", "GoogLeNet"]
    
    num_classes = [6,5,4,3,6,5,4,3]
    arg_outs = ["SixGoogLeNet","FiveGooLeNet","FourGoogLeNet","ThreeGoogLeNet",
                "SixAlexNet","FiveAlexNet","FourAlexNet","ThreeAlexNet"]
    
    csv_name_1000 = ['SixLineClass_1000.csv','FiveLineClass_1000.csv','FourLineClass_1000.csv','ThreeLineClass_1000.csv']*2
    csv_name_100 = ['SixLineClass_100.csv','FiveLineClass_100.csv','FourLineClass_100.csv','ThreeLineClass_100.csv']*2
    
    arg_model = ["GoogLeNet" for i in range(0,4)] + ["AlexNet" for i in range(0,4)]
    
    for i in range(0, len(num_classes)):
        a1[1] = arg_outs[i]
        a1[-1] = arg_model[i]
        main(a1, num_classes[i], csv_name_100[i])
