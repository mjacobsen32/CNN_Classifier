from .run import run
from helper_functions.get_args import get_args
def main():
    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations', 'perimeter=pad']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations', 'edge-pad']))

    run(get_args(['--batch-size', '32', '--test-batch-size', '32', 
                            '--epochs', '50', '--lr', '0.00001', '--gamma', '0.7', 
                            '--log-interval', '100', '--num-classes', '2', '--optimizer', "Adam",
                            '--model', 'AlexNet', '--percent-data', '1', '--loss', 'CEL',
                            '--input-dimension', '224', '--lr-sched', 'ReduceLROnPlateau',
                            '--save-model', '--write-log-interval', '100', '--augmentations', 'black-pad']))

if __name__ == '__main__':
    main()