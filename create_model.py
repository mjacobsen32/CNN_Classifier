from model.new_model import Model
from model.args import parser
from custom_datasets import phytoplankton, threeLines
from constants import planktonConstants, scatterPlotConstants
from torchvision import transforms

def main():
    args = parser.parse_args()
    c = args.constants

    if c == 'scatterPlotConstants':
        from custom_datasets.threeLines import ThreeLinesScatterPlot as Dataset
        import constants.scatterPlotConstants as constants
    elif c == 'planktonConstants':
        from custom_datasets.phytoplankton import PhytoplanktonDataset as Dataset
        import constants.planktonConstants as constants

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
    model.loss_func(args.loss)
    model.optimization_func(args.optim, args.learning_rate)
    model.lr_sched(args.scheduler, args.gamma, args.step_size)
    model.dataset_lengths(train=0.80, validation=0.20, test=0.0)
    model.subsets(modelLoaderKwargs)
    model.train(args.epochs)



if __name__ == '__main__':
    main()