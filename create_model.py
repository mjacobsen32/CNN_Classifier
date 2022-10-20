from model.new_model import Model
from model.args import parser
from custom_datasets import phytoplankton, threeLines
from constants import scatterPlotConstants, planktonConstants
from torchvision import transforms
from model.plotting import Plots
import os, shutil

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

    dataset = Dataset(constants.csv_path, constants.images_path, tf, None)
    model = Model(args.device, args.model, constants.num_classes, dataset)
    model.set_loss_func(args.loss)
    model.set_optimization_func(args.optim, args.learning_rate)
    model.set_lr_sched(args.scheduler, args.gamma, args.step_size)
    model.set_subset_indices(train=0.7, validation=0.15, test=0.15)
    model.set_subsets(modelLoaderKwargs)
    model.write_parameters(os.path.join(constants.output_folder, args.output_folder),
                            args.model, args.loss, args.optim, args.scheduler)
    model.train(args.epochs)
    model.test()
    model.results(os.path.join(constants.output_folder, args.output_folder))

    Plots.plot_val_acc_loss(model.loss_list, 
                            model.validation_accuracy_list, 
                            os.path.join(constants.output_folder, 
                                            args.output_folder, 
                                            args.output_folder+'.png'))

    
    if args.save_model:
        model.save_model(args.output_folder)



if __name__ == '__main__':
    main()