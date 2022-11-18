import torch
import torchvision.models as models
from model.timeFunctions import timer_wrapper

class BaseModel:
    def __init__(self, device_str, model_name, num_classes, ds):
        use_cuda = (torch.cuda.is_available() and device_str == 'cuda')
        self.device = torch.device("cuda" if use_cuda else "cpu") # Either 'Cuda' or 'CPU' (GPU or CPU usage)
        self.model = self.get_model(model_name, num_classes) # Creating model: To be loaded from model dict
                                                    # or created from scratch with seed 
                                                    # via inference model or new model
        self.dataset = ds                            # custom dataset to use
        self.num_classes = num_classes

    def get_model(self, model_name, num_classes):
        if model_name == "AlexNet":
            return models.alexnet(num_classes=num_classes).to(self.device)
        elif model_name == 'GoogLeNet':
            return models.googlenet(num_classes=num_classes, aux_logits=False, init_weights=True).to(self.device)
        elif model_name == 'ResNet18':
            return models.resnet18(num_classes=num_classes).to(self.device)
        elif model_name == 'VGG13':
            return models.vgg13(num_classes=num_classes).to(self.device)
        return "ERROR_LOADING_MODEL"