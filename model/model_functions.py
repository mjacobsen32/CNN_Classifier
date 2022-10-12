import torch.optim as optim
import torch
import torchvision.models as models

def get_model(model_name, num_classes, device):
    if model_name == "AlexNet":
        nn = models.alexnet(num_classes=num_classes).to(device)
    elif model_name == 'DConvNetV2':
        nn = DConvNetV2(num_classes).to(device)
    elif model_name == 'squeezenet1_1':
        nn = models.squeezenet1_1(num_classes=num_classes).to(device)
    elif model_name == 'InceptionV3':
        nn = models.inception_v3(num_classes=num_classes).to(device)
    elif model_name == 'GoogleNet':
        nn = models.googlenet(num_classes=num_classes,aux_logits=False).to(device)
    elif model_name == 'MobileNet_small':
        nn = models.mobilenet_v3_small(num_classes=num_classes).to(device)
    elif model_name == 'MobileNet_large':
        nn = models.mobilenet_v3_large(num_classes=num_classes).to(device)
    elif model_name == 'EfficientNet_B0':
        nn = models.efficientnet_b0(num_classes=num_classes).to(device)
    elif model_name == 'EfficientNet_B3':
        nn = models.efficientnet_b3(num_classes=num_classes).to(device)
    elif model_name == 'EfficientNet_B7':
        nn = models.efficientnet_b7(num_classes=num_classes).to(device)
    elif model_name == 'DenseNet169':
        nn = models.densenet169(num_classes=num_classes).to(device)
    elif model_name == 'ResNet50':
        nn = models.resnet50(num_classes=num_classes).to(device)
    elif model_name == 'VGG13':
        nn = models.vgg13(num_classes=num_classes).to(device)
    elif model_name == 'ConvNext_tiny':
        nn = models.convnext_tiny(num_classes=num_classes)
    elif model_name == 'ConvNext_small':
        nn = models.convnext_small(num_classes=num_classes)
    elif model_name == 'ConvNext_base':
        nn = models.convnext_base(num_classes=num_classes)
    elif model_name == 'ConvNext_large':
        nn = models.convnext_large(num_classes=num_classes)
    return(nn)


def get_optimizer(opt_name, lr, params):
    if opt_name == "AdaDelta":
        return(optim.Adadelta(params))
    elif opt_name == "Adam":
        return(optim.Adam(params, lr))
    elif opt_name == "SGD":
        return(optim.SGD(params, lr))
    elif opt_name == "RProp":
        return(optim.Rprop(params))
    elif opt_name == "AdaGrad":
        return(optim.Adagrad(params))


def get_lr_sched(sched_name, optimizer, gamma, step_size):
    if sched_name == "Plat":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.1, 
            patience=5, 
            verbose=True)
    elif sched_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma, 
            verbose=False)
    return(scheduler)


def get_loss_fn(loss_name, class_weights):
    if loss_name == "CEL":
        return(torch.nn.CrossEntropyLoss(weight=class_weights,
                                         reduction='sum'))